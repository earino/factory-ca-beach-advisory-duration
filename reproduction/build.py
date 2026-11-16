import hashlib
import heapq
import pathlib
import re

import numpy as np
import pandas as pd

from source_io import fetch

ADVISORY_URL = 'https://publish.data.ca.gov/dataset/b9c8ce91-40ff-4ad3-8164-bc17c46afb44/resource/d5cd6a23-829c-426d-a63e-689a55a3db9c/download/beach-advisories.csv'
STATION_URL = 'https://publish.data.ca.gov/dataset/b9c8ce91-40ff-4ad3-8164-bc17c46afb44/resource/98e628ff-d012-4982-ad32-b9f9ad8ab524/download/beach-monitoring-stations.csv'

FEATURES = ['advisory_class', 'start_month', 'start_day_of_week', 'county', 'beach_id', 'prior_postings_365d', 'prior_mean_closed_duration_days_365d', 'days_since_prior_posting', 'station_count', 'latitude', 'longitude', 'monitoring_agency']
OUT_COLUMNS = ['advisory_id', 'advisory_start_date', 'persists_7d'] + FEATURES

START_ALIASES = ['DateofAdvisory', 'Date of Advisory', 'Date_Of_Advisory', 'AdvisoryDate', 'advisory_start_date', 'start date', 'date posted', 'posting date']
END_ALIASES = ['DateOpened', 'Date Opened', 'Date_Opened', 'Date Closed', 'date closed', 'removal date', 'end date']
REPORTED_DURATION_ALIASES = ['DURATION CALCULATED', 'Duration Calculated', 'DURATION CALCULATED (DAYS)', 'DURATION_CALCULATED', 'DurationCalculated', 'duration calculated']
SPATIAL_EXTENT_ALIASES = ['EndofExtent', 'End of Extent', 'End_Extent', 'StartofExtent', 'Start of Extent', 'EndOfExtentType']
BEACH_ID_ALIASES = ['BeachName_id', 'BeachNameID', 'BeachNameId', 'Beach_Name_id', 'Beach Name id', 'beach_id', 'beach id', 'site_id', 'site id', 'beach_code', 'beach code', 'location_id', 'location id', 'Station_BeachName_id', 'StationBeachName_id']
BEACH_NAME_ALIASES = ['Beach_Name', 'Beach Name', 'beach_name', 'beach name', 'beach', 'site_name', 'site name', 'Station_Name', 'Station Name', 'station_name', 'location', 'location name']
COUNTY_ALIASES = ['County', 'CountyName', 'County Name', 'Beach_County', 'county_name', 'county name', 'countyorcity', 'county or city', 'jurisdiction', 'Station_County', 'Station County']
AGENCY_ALIASES = ['Beach_AgencyName', 'Agency_Name', 'AgencyName', 'Agency_Code', 'Station_Agency_id', 'Station_Agency', 'station_agency', 'agency', 'monitoring_agency', 'monitoring agency', 'submitting_agency', 'submitting agency', 'local_health_agency', 'health agency']
LAT_ALIASES = ['Station_UpperLat', 'Beach_UpperLat', 'Station_LowerLat', 'Beach_LowerLat', 'Station_Latitude', 'Station Latitude', 'Latitude', 'latitude', 'lat', 'latitude_deg', 'y_coord', 'y coordinate']
LON_ALIASES = ['Station_UpperLon', 'Beach_UpperLon', 'Station_LowerLon', 'Beach_LowerLon', 'Station_Longitude', 'Station Longitude', 'Longitude', 'longitude', 'lon', 'lng', 'longitude_deg', 'x_coord', 'x coordinate']
CLASS_ALIASES = ['AdvisoryType', 'Advisory Type', 'Advisory_Type', 'advisory_type', 'advisory type', 'advisory_class', 'advisory class', 'type of advisory', 'advisorycategory', 'advisory category', 'postingtype', 'posting type', 'type']


def _norm(value):
    return re.sub('[^a-z0-9]+', '', str(value).lower())


def _columns_for(df, aliases):
    cmap = {}
    for column in df.columns:
        cmap.setdefault(_norm(column), column)
    out = []
    for alias in aliases:
        key = _norm(alias)
        column = cmap.get(key)
        if column is not None and column not in out:
            out.append(column)
    return out


def _sample_values(series, limit=5):
    out = []
    for value in series.dropna().astype('string').head(200):
        text = str(value).strip()
        if text and text not in out:
            out.append(text[:80])
        if len(out) >= limit:
            break
    return out


def _raise_missing(label, headers, aliases, member, detail=None):
    header_list = list(headers) if headers is not None else []
    alias_list = list(aliases) if aliases is not None else []
    msg = 'Missing required source field for ' + str(label) + '; selected member ' + str(member) + '; observed headers ' + repr(header_list) + '; missing names ' + repr(alias_list)
    if detail:
        msg = msg + '; ' + str(detail)
    raise AssertionError(msg[:4000])


def _find(df, aliases, label, member, required=True):
    cols = _columns_for(df, aliases)
    if cols:
        return cols[0]
    if required:
        _raise_missing(label, df.columns, aliases, member)
    return None


def _clean_text(series):
    out = series.astype('string').str.strip()
    out = out.mask(out.eq(''), pd.NA)
    out = out.mask(out.str.lower().isin(['nan', 'none', 'null', 'n/a', 'na']), pd.NA)
    return out


def _key_component(series):
    s = series.astype('string').str.lower().str.strip()
    s = s.mask(s.str.endswith('.0', na=False), s.str.slice(0, -2))
    s = s.str.replace('[^a-z0-9]+', '', regex=True)
    return s.mask(s.eq(''), pd.NA)


def _make_beach_keys(source_df, id_col, name_col, county_col):
    idx = source_df.index
    if id_col is None:
        id_key = pd.Series(pd.NA, index=idx, dtype='string')
    else:
        id_key = _key_component(_clean_text(source_df[id_col]))
    if name_col is None:
        name_key = pd.Series(pd.NA, index=idx, dtype='string')
    else:
        name_key = _key_component(_clean_text(source_df[name_col]))
    if county_col is None:
        county_key = pd.Series(pd.NA, index=idx, dtype='string')
    else:
        county_key = _key_component(_clean_text(source_df[county_col]))
    name_only_key = name_key
    name_county_key = name_key.str.cat(county_key, sep='|', na_rep='')
    name_county_key = name_county_key.mask(name_key.isna(), pd.NA)
    beach_key = id_key.where(id_key.notna(), name_county_key)
    return id_key, name_county_key, name_only_key, beach_key


def _parse_dates(series):
    s = series.astype('string').str.strip()
    s = s.mask(s.eq(''), pd.NA)
    low = s.str.lower()
    s = s.mask(low.isin(['nan', 'none', 'null', 'n/a', 'na', 'ongoing', 'open', 'unknown', '-', '--']), pd.NA)
    try:
        parsed = pd.to_datetime(s, errors='coerce', utc=True, format='mixed')
    except (TypeError, ValueError):
        try:
            parsed = pd.to_datetime(s, errors='coerce', utc=True)
        except (TypeError, ValueError):
            parsed = pd.Series(pd.NaT, index=s.index)
    if isinstance(parsed.dtype, pd.DatetimeTZDtype):
        parsed = parsed.dt.tz_convert(None)
    parsed = parsed.dt.normalize()
    if parsed.notna().sum() == 0 and s.notna().any():
        vals = []
        for value in s:
            if pd.isna(value):
                vals.append(pd.NaT)
            else:
                try:
                    vals.append(pd.to_datetime(value, errors='coerce', utc=True))
                except Exception:
                    vals.append(pd.NaT)
        parsed = pd.Series(vals, index=s.index)
        if isinstance(parsed.dtype, pd.DatetimeTZDtype):
            parsed = parsed.dt.tz_convert(None)
        parsed = parsed.dt.normalize()
    nums = pd.to_numeric(s.str.replace(',', '', regex=False), errors='coerce')
    low_bound = pd.Timestamp('1980-01-01')
    high_bound = pd.Timestamp('2031-01-01')
    implausible = parsed.isna() | (parsed < low_bound) | (parsed >= high_bound)
    excel_mask = nums.between(20000, 80000) & implausible
    if excel_mask.any():
        parsed.loc[excel_mask] = pd.to_datetime(nums.loc[excel_mask], unit='D', origin='1899-12-30', errors='coerce')
    compact_mask = s.str.fullmatch('[0-9]{8}', na=False) & implausible
    if compact_mask.any():
        parsed.loc[compact_mask] = pd.to_datetime(s.loc[compact_mask], format='%Y%m%d', errors='coerce')
    parsed = parsed.where((parsed >= low_bound) & (parsed < high_bound), pd.NaT)
    return parsed


def _coalesce_dates(df, aliases, label, member, required=True):
    result = pd.Series(pd.NaT, index=df.index, dtype='datetime64[ns]')
    used = []
    cols = _columns_for(df, aliases)
    for col in cols:
        parsed = _parse_dates(df[col])
        if parsed.notna().any():
            fill = result.isna() & parsed.notna()
            result.loc[fill] = parsed.loc[fill]
            used.append(col)
    if required and result.notna().sum() == 0:
        detail = 'candidate columns ' + repr(cols)
        if cols:
            detail = detail + '; sample values ' + repr({col: _sample_values(df[col]) for col in cols[:3]})
        _raise_missing(label, df.columns, aliases, member, detail=detail)
    return result, used


def _parse_decimal_days(series):
    s = series.astype('string').str.strip()
    s = s.mask(s.eq(''), pd.NA)
    low = s.str.lower()
    s = s.mask(low.isin(['nan', 'none', 'null', 'n/a', 'na', 'open', 'ongoing', 'unknown', '-', '--']), pd.NA)
    s = s.str.replace(',', '', regex=False)
    return pd.to_numeric(s, errors='coerce')


def _coalesce_decimal_days(df, aliases, label, member, required=False):
    result = pd.Series(np.nan, index=df.index, dtype='float64')
    used = []
    cols = _columns_for(df, aliases)
    for col in cols:
        vals = _parse_decimal_days(df[col]).replace([np.inf, -np.inf], np.nan)
        if vals.notna().any():
            fill = result.isna() & vals.notna()
            result.loc[fill] = vals.loc[fill]
            used.append(col)
    if required and result.notna().sum() == 0:
        detail = 'candidate columns ' + repr(cols)
        if cols:
            detail = detail + '; sample values ' + repr({col: _sample_values(df[col]) for col in cols[:3]})
        _raise_missing(label, df.columns, aliases, member, detail=detail)
    return result, used


def _parse_numeric(series):
    s = series.astype('string').str.strip()
    s = s.mask(s.eq(''), pd.NA)
    vals = pd.to_numeric(s.str.replace(',', '', regex=False), errors='coerce')
    missing = vals.isna() & s.notna()
    if missing.any():
        for i in s.index[missing]:
            text = str(s.at[i])
            match = re.search('-?[0-9]+(?:[.][0-9]+)?', text)
            if match:
                try:
                    vals.at[i] = float(match.group(0))
                except Exception:
                    pass
    return vals


def _coalesce_numeric(df, aliases):
    result = pd.Series(np.nan, index=df.index, dtype='float64')
    used = []
    for col in _columns_for(df, aliases):
        vals = _parse_numeric(df[col]).replace([np.inf, -np.inf], np.nan)
        if vals.notna().any():
            fill = result.isna() & vals.notna()
            result.loc[fill] = vals.loc[fill]
            used.append(col)
    return result, used


def _coalesce_text(df, aliases, label, member, required=False):
    result = pd.Series(pd.NA, index=df.index, dtype='string')
    used = []
    for col in _columns_for(df, aliases):
        cleaned = _clean_text(df[col])
        if cleaned.notna().any():
            fill = result.isna() & cleaned.notna()
            result.loc[fill] = cleaned.loc[fill]
            used.append(col)
    if required and result.notna().sum() == 0:
        _raise_missing(label, df.columns, aliases, member, detail='candidate columns ' + repr(used))
    return result, used


def _first_nonempty(series):
    for value in series:
        if pd.notna(value):
            text = str(value).strip()
            if text:
                return text
    return pd.NA


def _aggregate_stations(st, key_col):
    columns = ['station_count', 'latitude', 'longitude', 'monitoring_agency', 'county']
    if key_col is None:
        return pd.DataFrame(columns=columns)
    sub = st.dropna(subset=[key_col]).copy()
    if sub.empty:
        return pd.DataFrame(columns=columns)
    return sub.groupby(key_col, dropna=True, sort=False).agg(
        station_count=(key_col, 'size'),
        latitude=('_lat_st', 'mean'),
        longitude=('_lon_st', 'mean'),
        monitoring_agency=('_agency_st', _first_nonempty),
        county=('_county_st', _first_nonempty),
    )


def _map_station_metric(keys, agg, column, index):
    if agg is None or not hasattr(agg, 'empty') or agg.empty or column not in agg.columns:
        return pd.Series(np.nan, index=index, dtype='float64')
    return keys.map(agg[column])


def _map_numeric_station_metric(keys1, agg1, keys2, agg2, keys3, agg3, column, index):
    out = _map_station_metric(keys1, agg1, column, index)
    out2 = _map_station_metric(keys2, agg2, column, index)
    out = out.where(out.notna(), out2)
    out3 = _map_station_metric(keys3, agg3, column, index)
    out = out.where(out.notna(), out3)
    return out


def _map_text_station_metric(keys1, agg1, keys2, agg2, keys3, agg3, column, index):
    out = _map_station_metric(keys1, agg1, column, index).astype('object')
    out2 = _map_station_metric(keys2, agg2, column, index).astype('object')
    out = out.where(out.notna(), out2)
    out3 = _map_station_metric(keys3, agg3, column, index).astype('object')
    out = out.where(out.notna(), out3)
    return out.astype('string')


def _combine_text(primary, fallback):
    p = primary.astype('object')
    f = fallback.astype('object')
    out = p.where(p.notna(), f)
    out = out.where(out.notna(), 'UNKNOWN')
    return out.astype('string')


def _split_name(starts):
    out = pd.Series(pd.NA, index=starts.index, dtype='string')
    out[(starts >= pd.Timestamp('2005-01-01')) & (starts < pd.Timestamp('2016-01-01'))] = 'train'
    out[(starts >= pd.Timestamp('2016-01-01')) & (starts < pd.Timestamp('2020-01-01'))] = 'eval'
    out[(starts >= pd.Timestamp('2020-01-01')) & (starts < pd.Timestamp('2024-01-01'))] = 'holdout'
    return out


def _embargo_mask(starts):
    mask = pd.Series(False, index=starts.index)
    for boundary in [pd.Timestamp('2016-01-01'), pd.Timestamp('2020-01-01'), pd.Timestamp('2024-01-01')]:
        mask = mask | ((starts >= boundary - pd.Timedelta(days=7)) & (starts < boundary))
    return mask


def _add_prior_features(base, hist, train_labels):
    n = len(base)
    counts = np.zeros(n, dtype=np.int64)
    days = np.full(n, np.nan)
    means = np.full(n, np.nan)
    worst_gap = None
    hist_by = {key: group.sort_values('advisory_start_date') for key, group in hist.groupby('_beach_key', sort=False)}
    train_by = {key: group.sort_values('_end_boundary') for key, group in train_labels.groupby('_beach_key', sort=False)}
    for beach, idx in base.groupby('_beach_key', sort=False).groups.items():
        idx = np.asarray(idx)
        h = hist_by.get(beach)
        if h is None:
            starts = np.array([], dtype='datetime64[ns]')
        else:
            starts = h['advisory_start_date'].to_numpy(dtype='datetime64[ns]')
        cur = base.loc[idx, 'advisory_start_date'].to_numpy(dtype='datetime64[ns]')
        left = np.searchsorted(starts, cur - np.timedelta64(365, 'D'), side='left')
        right = np.searchsorted(starts, cur, side='left')
        counts[idx] = np.maximum(0, right - left)
        prev = right - 1
        ok = prev >= 0
        if ok.any():
            days[idx[ok]] = (cur[ok] - starts[prev[ok]]) / np.timedelta64(1, 'D')
        tr = train_by.get(beach)
        if tr is not None and len(tr):
            tr_ends = tr['_end_boundary'].to_numpy(dtype='datetime64[ns]')
            tr_starts = tr['advisory_start_date'].to_numpy(dtype='datetime64[ns]')
            tr_dur = tr['duration_days'].to_numpy(dtype='float64')
            pointer = 0
            active = []
            total = 0.0
            count = 0
            for i in idx:
                cur_i = np.datetime64(base.at[i, 'advisory_start_date'], 'ns')
                while pointer < len(tr_ends) and tr_ends[pointer] < cur_i:
                    heapq.heappush(active, (tr_starts[pointer], float(tr_dur[pointer])))
                    total = total + float(tr_dur[pointer])
                    count = count + 1
                    gap = float((tr_ends[pointer] - cur_i) / np.timedelta64(1, 'D'))
                    worst_gap = gap if worst_gap is None else max(worst_gap, gap)
                    pointer = pointer + 1
                threshold = cur_i - np.timedelta64(365, 'D')
                while active and active[0][0] < threshold:
                    _, duration = heapq.heappop(active)
                    total = total - duration
                    count = count - 1
                if count:
                    means[i] = total / count
    base['prior_postings_365d'] = counts.astype(int)
    base['prior_mean_closed_duration_days_365d'] = means
    base['days_since_prior_posting'] = days
    return base, {'max_end_minus_start_days': worst_gap}


def _stable_hash(value):
    return 'B' + hashlib.sha1(str(value).encode('utf-8')).hexdigest()[:16].upper()


def build(output: pathlib.Path, cache: pathlib.Path):
    output = pathlib.Path(output)
    cache = pathlib.Path(cache)
    advisory_path = pathlib.Path(fetch(ADVISORY_URL, cache, 'beach-advisories.csv'))
    station_path = pathlib.Path(fetch(STATION_URL, cache, 'beach-monitoring-stations.csv'))
    adv = pd.read_csv(advisory_path, encoding='utf-8-sig', low_memory=False)
    st = pd.read_csv(station_path, encoding='utf-8-sig', low_memory=False)

    class_col = _find(adv, CLASS_ALIASES, 'advisory class', advisory_path.name)
    start_date, start_used = _coalesce_dates(adv, START_ALIASES, 'advisory issue date', advisory_path.name, required=True)
    end_date, end_used = _coalesce_dates(adv, END_ALIASES, 'advisory removal date', advisory_path.name, required=True)
    reported_duration, reported_used = _coalesce_decimal_days(adv, REPORTED_DURATION_ALIASES, 'reported advisory duration', advisory_path.name)
    spatial_cols = _columns_for(adv, SPATIAL_EXTENT_ALIASES)

    beach_id_col = _find(adv, BEACH_ID_ALIASES, 'advisory beach identifier', advisory_path.name, required=False)
    beach_name_col = _find(adv, BEACH_NAME_ALIASES, 'advisory beach name', advisory_path.name, required=False)
    if beach_id_col is None and beach_name_col is None:
        _raise_missing('beach identifier or beach name', adv.columns, BEACH_ID_ALIASES + BEACH_NAME_ALIASES, advisory_path.name)
    county_col = _find(adv, COUNTY_ALIASES, 'advisory county', advisory_path.name, required=False)
    agency_col = _find(adv, AGENCY_ALIASES, 'advisory agency', advisory_path.name, required=False)
    lat_col = _find(adv, LAT_ALIASES, 'advisory latitude', advisory_path.name, required=False)
    lon_col = _find(adv, LON_ALIASES, 'advisory longitude', advisory_path.name, required=False)

    raw = pd.DataFrame(index=adv.index)
    raw['_row'] = np.arange(len(raw))
    raw['_class_raw'] = _clean_text(adv[class_col])
    raw['_class_norm'] = raw['_class_raw'].str.lower().str.replace('[^a-z0-9]+', ' ', regex=True).str.strip()
    raw['advisory_start_date'] = start_date
    raw['advisory_end_date'] = end_date
    raw['duration_days'] = (end_date - start_date).dt.days.astype('float64')
    raw['_reported_duration'] = reported_duration
    raw['_county_adv'], county_used = _coalesce_text(adv, COUNTY_ALIASES, 'advisory county', advisory_path.name, required=False)
    raw['_agency_adv'], agency_used = _coalesce_text(adv, AGENCY_ALIASES, 'advisory agency', advisory_path.name, required=False)
    raw['_lat_adv'], lat_used = _coalesce_numeric(adv, LAT_ALIASES)
    raw['_lon_adv'], lon_used = _coalesce_numeric(adv, LON_ALIASES)
    keys = _make_beach_keys(adv, beach_id_col, beach_name_col, county_col)
    raw['_beach_id_key'] = keys[0]
    raw['_name_county_key'] = keys[1]
    raw['_name_only_key'] = keys[2]
    raw['_beach_key'] = keys[3]

    class_text = raw['_class_norm'].fillna('')
    posting_mask = class_text.str.contains('posting', na=False) & ~class_text.str.contains('permanent|closure|rain', na=False)
    class_counts = {str(key): int(value) for key, value in raw['_class_raw'].value_counts(dropna=False).items()}
    permanent_rows = int(class_text.str.contains('permanent', na=False).sum())
    posting_rows = int(posting_mask.sum())

    missing_start = int((posting_mask & raw['advisory_start_date'].isna()).sum())
    missing_end = int((posting_mask & raw['advisory_end_date'].isna()).sum())
    missing_duration = int((posting_mask & raw['advisory_start_date'].notna() & raw['advisory_end_date'].notna() & raw['duration_days'].isna()).sum())
    negative = int((posting_mask & raw['duration_days'].notna() & (raw['duration_days'] < 0)).sum())
    outside_window = int((posting_mask & raw['advisory_end_date'].notna() & raw['advisory_start_date'].notna() & ((raw['advisory_start_date'] < pd.Timestamp('2005-01-01')) | (raw['advisory_start_date'] >= pd.Timestamp('2024-01-01')))).sum())

    keyable = posting_mask & raw['advisory_start_date'].notna() & raw['_beach_key'].notna()
    date_part = raw['advisory_start_date'].dt.strftime('%Y-%m-%d')
    composite = raw['_beach_key'].astype('string').fillna('') + '|' + date_part.fillna('') + '|' + raw['_class_norm'].fillna('')
    keyed = composite.loc[keyable]
    duplicate_groups = int(keyed.nunique())
    duplicate_extra = int(keyed.duplicated().sum())
    keep_mask = pd.Series(False, index=raw.index)
    keep_mask.loc[keyable] = ~keyed.duplicated(keep='first')
    differing_duplicates = 0
    if duplicate_extra:
        grp = raw.loc[keyable].assign(_composite=keyed)
        nun = grp.groupby('_composite')['duration_days'].nunique(dropna=True)
        differing_duplicates = int((nun > 1).sum())

    valid = keep_mask & raw['advisory_start_date'].notna() & raw['duration_days'].notna() & (raw['duration_days'] >= 0) & (raw['advisory_start_date'] >= pd.Timestamp('2005-01-01')) & (raw['advisory_start_date'] < pd.Timestamp('2024-01-01'))
    base = raw.loc[valid].copy()
    if base.empty:
        detail = 'class counts ' + repr(class_counts) + '; start candidates ' + repr(start_used) + '; end candidates ' + repr(end_used) + '; posting rows ' + str(posting_rows) + '; missing start ' + str(missing_start) + '; missing end ' + str(missing_end) + '; negative ' + str(negative)
        raise AssertionError(('No includeable posting rows after date and duration filters; selected member ' + advisory_path.name + '; observed headers ' + repr(list(adv.columns)) + '; ' + detail)[:4000])
    if base['_beach_key'].isna().any():
        _raise_missing('beach identifier or beach name for included posting rows', adv.columns, BEACH_ID_ALIASES + BEACH_NAME_ALIASES, advisory_path.name, detail='missing beach keys ' + str(int(base['_beach_key'].isna().sum())) + '; included ' + str(len(base)))
    base['_split'] = _split_name(base['advisory_start_date'])
    embargo = _embargo_mask(base['advisory_start_date'])
    embargo_count = int(embargo.sum())
    base = base.loc[~embargo].copy()
    if base.empty:
        raise AssertionError('No rows remain after split boundary embargo; selected member ' + advisory_path.name)
    if base['_split'].isna().any():
        raise AssertionError('Rows outside declared half-open split windows remain after filtering; selected member ' + advisory_path.name)
    base = base.sort_values(['_beach_key', 'advisory_start_date', 'duration_days', '_row'], kind='mergesort').reset_index(drop=True)

    st_id_col = _find(st, BEACH_ID_ALIASES, 'station beach identifier', station_path.name, required=False)
    st_name_col = _find(st, BEACH_NAME_ALIASES, 'station beach name', station_path.name, required=False)
    if st_id_col is None and st_name_col is None:
        _raise_missing('station beach identifier or beach name', st.columns, BEACH_ID_ALIASES + BEACH_NAME_ALIASES, station_path.name)
    st_county_col = _find(st, COUNTY_ALIASES, 'station county', station_path.name, required=False)
    st_agency_col = _find(st, AGENCY_ALIASES, 'station agency', station_path.name, required=False)
    st_lat_col = _find(st, LAT_ALIASES, 'station latitude', station_path.name, required=False)
    st_lon_col = _find(st, LON_ALIASES, 'station longitude', station_path.name, required=False)
    if county_col is None and st_county_col is None:
        _raise_missing('county', adv.columns, COUNTY_ALIASES, advisory_path.name, detail='station headers ' + repr(list(st.columns)))
    if agency_col is None and st_agency_col is None:
        _raise_missing('monitoring agency', adv.columns, AGENCY_ALIASES, advisory_path.name, detail='station headers ' + repr(list(st.columns)))
    if lat_col is None and st_lat_col is None:
        _raise_missing('latitude', adv.columns, LAT_ALIASES, advisory_path.name, detail='station headers ' + repr(list(st.columns)))
    if lon_col is None and st_lon_col is None:
        _raise_missing('longitude', adv.columns, LON_ALIASES, advisory_path.name, detail='station headers ' + repr(list(st.columns)))

    st_keys = _make_beach_keys(st, st_id_col, st_name_col, st_county_col)
    st['_st_id_key'] = st_keys[0]
    st['_st_name_county_key'] = st_keys[1]
    st['_st_name_only_key'] = st_keys[2]
    st['_county_st'], station_county_used = _coalesce_text(st, COUNTY_ALIASES, 'station county', station_path.name, required=False)
    st['_agency_st'], station_agency_used = _coalesce_text(st, AGENCY_ALIASES, 'station agency', station_path.name, required=False)
    st['_lat_st'], station_lat_used = _coalesce_numeric(st, LAT_ALIASES)
    st['_lon_st'], station_lon_used = _coalesce_numeric(st, LON_ALIASES)
    id_agg = _aggregate_stations(st, '_st_id_key')
    name_county_agg = _aggregate_stations(st, '_st_name_county_key')
    name_only_agg = _aggregate_stations(st, '_st_name_only_key')
    idx = base.index
    station_count = _map_numeric_station_metric(base['_beach_id_key'], id_agg, base['_name_county_key'], name_county_agg, base['_name_only_key'], name_only_agg, 'station_count', idx)
    station_lat = _map_numeric_station_metric(base['_beach_id_key'], id_agg, base['_name_county_key'], name_county_agg, base['_name_only_key'], name_only_agg, 'latitude', idx)
    station_lon = _map_numeric_station_metric(base['_beach_id_key'], id_agg, base['_name_county_key'], name_county_agg, base['_name_only_key'], name_only_agg, 'longitude', idx)
    station_agency = _map_text_station_metric(base['_beach_id_key'], id_agg, base['_name_county_key'], name_county_agg, base['_name_only_key'], name_only_agg, 'monitoring_agency', idx)
    station_county = _map_text_station_metric(base['_beach_id_key'], id_agg, base['_name_county_key'], name_county_agg, base['_name_only_key'], name_only_agg, 'county', idx)
    base['station_count'] = station_count.round().astype('Int64')
    base['latitude'] = station_lat.where(station_lat.notna(), base['_lat_adv'])
    base['longitude'] = station_lon.where(station_lon.notna(), base['_lon_adv'])
    base['monitoring_agency'] = _combine_text(base['_agency_adv'], station_agency)
    base['county'] = _combine_text(base['_county_adv'], station_county)

    hist = raw.loc[keep_mask].copy()
    train_labels = base.loc[base['_split'].eq('train') & base['duration_days'].notna() & base['advisory_end_date'].notna()].copy()
    train_labels['_end_boundary'] = train_labels['advisory_end_date']
    train_labels = train_labels.loc[train_labels['_end_boundary'] < pd.Timestamp('2016-01-01')].copy()
    base, prior_stats = _add_prior_features(base, hist, train_labels)

    base['advisory_class'] = 'posting'
    base['start_month'] = base['advisory_start_date'].dt.month.astype(int)
    base['start_day_of_week'] = base['advisory_start_date'].dt.dayofweek.astype(int)
    base['beach_id'] = base['_beach_key'].map(_stable_hash)
    duration_text = base['duration_days'].map(lambda value: format(float(value), '.6f'))
    canonical = base['_beach_key'].astype(str) + '|' + base['advisory_start_date'].dt.strftime('%Y-%m-%d') + '|' + duration_text + '|' + base['advisory_class'].astype(str)
    base['_id_hash'] = canonical.map(lambda value: hashlib.sha1(value.encode('utf-8')).hexdigest()[:20].upper())
    base['_dup'] = base.groupby('_id_hash', sort=False).cumcount()
    base['advisory_id'] = base['_id_hash'] + base['_dup'].map(lambda value: '' if int(value) == 0 else '-' + str(int(value)))
    if base['duration_days'].isna().any():
        raise AssertionError('Included posting rows have missing duration after filtering; selected member ' + advisory_path.name)
    base['persists_7d'] = (base['duration_days'] >= 7.0).astype(int)

    checks = []
    def add_check(name, observed, expected, passed, kind=None):
        item = {'check': name, 'observed': observed, 'expected': expected, 'passed': bool(passed)}
        if kind:
            item['kind'] = kind
        checks.append(item)

    field_ok = (len(start_used) > 0 and len(end_used) > 0 and _norm(start_used[0]) == 'dateofadvisory' and _norm(end_used[0]) == 'dateopened' and (beach_id_col is not None or beach_name_col is not None) and (st_id_col is not None or st_name_col is not None) and (county_col is not None or st_county_col is not None) and (agency_col is not None or st_agency_col is not None) and (lat_col is not None or st_lat_col is not None) and (lon_col is not None or st_lon_col is not None))
    add_check('advisory source field mapping', 'class column ' + repr(class_col) + '; start candidates used ' + repr(start_used) + '; end candidates used ' + repr(end_used) + '; beach id column ' + repr(beach_id_col) + '; beach name column ' + repr(beach_name_col) + '; advisory headers ' + repr(list(adv.columns)) + '; station headers ' + repr(list(st.columns)), 'AdvisoryType gives the class, DateofAdvisory gives the issue date, DateOpened gives the removal date, and beach identity plus station inventory join fields are present', field_ok)

    add_check('advisory issue date field used', 'start candidates used ' + repr(start_used) + '; null issue dates among posting rows ' + str(missing_start) + '; posting rows ' + str(posting_rows), 'The time anchor is the advisory issue date column (DateofAdvisory), not the removal date column', len(start_used) > 0 and _norm(start_used[0]) == 'dateofadvisory' and missing_start == 0)

    add_check('advisory removal date field used', 'end candidates used ' + repr(end_used) + '; null removal dates among posting rows ' + str(missing_end) + '; posting rows ' + str(posting_rows), 'The advisory removal date column (DateOpened) supplies the end of the episode and is withheld from features', len(end_used) > 0 and _norm(end_used[0]) == 'dateopened')

    spatial_seen = {col: _sample_values(adv[col]) for col in spatial_cols[:3]}
    extent_not_used = len(set(_norm(c) for c in reported_used) & set(_norm(c) for c in spatial_cols)) == 0
    add_check('spatial extent column excluded from duration', 'spatial columns seen ' + repr(spatial_cols) + '; sample ' + repr(spatial_seen) + '; duration candidates used ' + repr(reported_used) + '; duration source is the removal date minus the issue date expressed in whole days', 'EndofExtent and related extent columns are spatial coordinate offsets (the publisher tags EndOfExtentType as a location type), so they must never be treated as day durations', extent_not_used and len(end_used) > 0)

    both = reported_duration.notna() & raw['duration_days'].notna()
    agree = (reported_duration - raw['duration_days']).abs() <= 1.0
    agree_count = int((both & agree).sum())
    agree_share = float(agree_count / max(int(both.sum()), 1))
    cross_ok = int(both.sum()) > 0 and agree_share >= 0.90
    add_check('computed duration agrees with reported duration', 'reported duration candidates ' + repr(reported_used) + '; rows with both values ' + str(int(both.sum())) + '; agreeing within one day ' + str(agree_count) + '; agreement share ' + format(agree_share, '.4f') + '; source headers ' + repr(list(adv.columns)), 'The publisher duration column (DURATION CALCULATED) matches the removal date minus the issue date in whole calendar days for the large majority of rows', cross_ok)

    dur = base['duration_days'].astype(float)
    unit_ok = len(dur) > 0 and np.isfinite(dur.to_numpy()).all() and bool((dur >= 0).all())
    add_check('label duration unit and seven-day threshold', 'rows ' + str(len(dur)) + '; min days ' + format(float(dur.min()), '.6f') + '; max days ' + format(float(dur.max()), '.6f') + '; threshold days 7; threshold seconds 604800; duration frame is whole calendar days from issue date to removal date', 'Duration is the whole-calendar-day difference between the advisory issue date and the advisory removal date, and seven days equals 604800 seconds', unit_ok, kind='unit')

    add_check('posting advisory class present', 'class counts ' + repr(class_counts) + '; posting rows ' + str(posting_rows) + '; rows whose class text mentions permanent ' + str(permanent_rows), 'The posting class appears; closure, rain advisory and any permanent-tagged class are counted and excluded, and non-posting records are not required absent', posting_rows > 0)

    retained_ok = len(base) > 0 and int(base['advisory_start_date'].isna().sum()) == 0 and int(base['duration_days'].isna().sum()) == 0 and int((base['duration_days'] < 0).sum()) == 0
    add_check('retained rows have coherent issue date and duration', 'posting rows ' + str(posting_rows) + '; missing issue date ' + str(missing_start) + '; missing removal date ' + str(missing_end) + '; missing duration ' + str(missing_duration) + '; negative duration ' + str(negative) + '; outside window ' + str(outside_window) + '; included ' + str(len(base)), 'Retained rows have a parsed issue date, a parsed removal date on or after the issue date, and a nonnegative whole-day duration; open and negative records are excluded', retained_ok)

    add_check('duplicate beach-start-class rows removed', 'candidate posting rows ' + str(int(keyable.sum())) + '; distinct beach-start-class groups ' + str(duplicate_groups) + '; duplicate extra rows ' + str(duplicate_extra) + '; duplicate groups with differing durations ' + str(differing_duplicates) + '; remaining duplicates after keeping the first source record ' + str(int(base.duplicated(subset=['_beach_key', 'advisory_start_date', '_class_norm']).sum())), 'At most one record per beach, issue date and advisory class survives, kept deterministically as the earliest source row, so prior counts are not inflated by repeated episode rows', duplicate_groups == duplicate_groups and int(base.duplicated(subset=['_beach_key', 'advisory_start_date', '_class_norm']).sum()) == 0)

    split_counts = {str(key): int(value) for key, value in base['_split'].value_counts().items()}
    split_ok = all(split_counts.get(name, 0) > 0 for name in ['train', 'eval', 'holdout'])
    add_check('declared split windows populated', 'split counts ' + repr(split_counts), 'Nonempty chronological train, eval and holdout windows', split_ok)

    positive_counts = {str(key): int(value) for key, value in base.groupby('_split')['persists_7d'].sum().items()}
    positive_ok = all(positive_counts.get(name, 0) > 0 for name in ['train', 'eval', 'holdout'])
    add_check('positive labels present in every split', 'positive counts ' + repr(positive_counts), 'Each split contains at least one posting that stayed in force seven days or longer', positive_ok)

    duplicate_ids = int(base['advisory_id'].duplicated().sum())
    add_check('stable advisory IDs unique', 'duplicate advisory identifier count ' + str(duplicate_ids), 'Zero duplicate advisory identifier values', duplicate_ids == 0)

    missing_feature_columns = [name for name in FEATURES if name not in base.columns]
    empty_features = [name for name in FEATURES if name in base.columns and base[name].isna().all()]
    post_hoc = {'endofextent', 'duration calculated', 'durationcalculated', 'duration_days', 'dateopened'}
    leak = sorted(set(FEATURES) & post_hoc)
    feature_ok = len(missing_feature_columns) == 0 and len(empty_features) == 0 and len(leak) == 0
    add_check('declared feature columns resolved', 'missing ' + repr(missing_feature_columns) + '; empty ' + repr(empty_features) + '; post outcome leak ' + repr(leak) + '; station headers ' + repr(list(st.columns)), 'Every declared feature column exists, none is entirely missing, and no removal date, duration or spatial extent field is shipped as a feature', feature_ok)

    station_matched = int(base['station_count'].notna().sum())
    add_check('station inventory context coverage', 'matched ' + str(station_matched) + ' of ' + str(len(base)), 'At least one station match; unmatched rows retain missing station fields rather than zero', station_matched > 0)

    county_known = int(base['county'].astype('string').ne('UNKNOWN').sum())
    add_check('county feature coverage', 'known ' + str(county_known) + ' of ' + str(len(base)), 'At least one resolved county from the advisory record or the station inventory; missing values are the explicit UNKNOWN category', county_known > 0)

    agency_known = int(base['monitoring_agency'].astype('string').ne('UNKNOWN').sum())
    add_check('monitoring agency coverage', 'known ' + str(agency_known) + ' of ' + str(len(base)), 'At least one resolved monitoring agency from the advisory record or the station inventory; missing values are the explicit UNKNOWN category', agency_known > 0)

    mean_known = int(base['prior_mean_closed_duration_days_365d'].notna().sum())
    train_mean_known = int(base.loc[base['_split'].eq('train'), 'prior_mean_closed_duration_days_365d'].notna().sum())
    train_label_count = int(len(train_labels))
    max_train_end = train_labels['_end_boundary'].max() if train_label_count else pd.NaT
    label_window_ok = train_label_count > 0 and pd.notna(max_train_end) and max_train_end < pd.Timestamp('2016-01-01')
    add_check('training-window prior duration coverage', 'known ' + str(mean_known) + ' of ' + str(len(base)) + '; train known ' + str(train_mean_known) + '; training labels used ' + str(train_label_count) + '; latest training removal date ' + str(max_train_end), 'Prior mean duration uses only training records whose recorded removal date is before the scored issue date and before 2016-01-01; missingness is retained where no qualifying training record exists', train_mean_known > 0 and label_window_ok)

    rate_sources = sorted(set(train_labels['_split'].astype(str)))
    gap_value = prior_stats.get('max_end_minus_start_days')
    gap_ok = gap_value is None or float(gap_value) <= -1.0
    add_check('training-only causal rate labels', 'rate label splits ' + repr(rate_sources) + '; latest training removal date ' + str(max_train_end) + '; latest removal date minus scored issue date in days ' + str(gap_value), 'Prior duration rates use training-split records only, and each contributing removal date falls strictly before the scored issue date and before the training window end', rate_sources == ['train'] and label_window_ok and gap_ok)

    remaining_embargo = int(_embargo_mask(base['advisory_start_date']).sum())
    add_check('split boundary embargo', 'dropped ' + str(embargo_count) + '; remaining in embargo ' + str(remaining_embargo), 'Records starting in the seven days before 2016-01-01, 2020-01-01 or 2024-01-01 are dropped; remaining rows fall in half-open windows', remaining_embargo == 0)

    start_min = base['advisory_start_date'].min()
    start_max = base['advisory_start_date'].max()
    range_ok = start_min >= pd.Timestamp('2005-01-01') and start_max < pd.Timestamp('2024-01-01')
    add_check('included start dates in frozen range', 'min ' + str(start_min) + '; max ' + str(start_max), 'All included issue dates fall from 2005-01-01 through 2023-12-31', range_ok)

    unique_labels = sorted(set(int(value) for value in base['persists_7d'].unique()))
    label_ok = base['persists_7d'].notna().all() and set(unique_labels).issubset({0, 1})
    add_check('binary target construction', 'unique target values ' + repr(unique_labels) + '; positives ' + str(int(base['persists_7d'].sum())) + '; rows ' + str(len(base)), 'Target contains only 0 and 1 with no missing values and no record with an unknown removal date is treated as negative', label_ok)

    out = base[OUT_COLUMNS].copy()
    out_columns_ok = list(out.columns) == OUT_COLUMNS
    add_check('output column contract', 'observed columns ' + repr(list(out.columns)) + '; target header ' + repr('persists_7d') + '; declared features ' + repr(FEATURES), 'Output columns exactly match the declared identifier, time, target and feature names', out_columns_ok)

    failed = [check for check in checks if not check['passed']]
    assert not failed, 'Source checks failed: ' + repr(failed)[:4000]

    base = base.sort_values(['advisory_start_date', 'advisory_id'], kind='mergesort').reset_index(drop=True)
    out = base[OUT_COLUMNS].copy()
    out['advisory_start_date'] = out['advisory_start_date'].dt.strftime('%Y-%m-%dT%H:%M:%S.000')
    for split_name, relative in [('train', 'public/train.csv'), ('eval', 'public/eval.csv'), ('holdout', 'private/holdout.csv')]:
        frame = out.loc[base['_split'].eq(split_name)].copy()
        if frame.empty:
            raise AssertionError('Empty split after construction: ' + split_name)
        path = output / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False, na_rep='', lineterminator=chr(10))
    return checks
