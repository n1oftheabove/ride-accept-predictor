import pandas as pd
import numpy as np


def load_original_data():
    br = pd.read_csv("data/bookingRequests.csv")
    rr = pd.read_csv("data/rideRequests.csv")

    return pd.merge(
        br,
        rr,
        how="left",
        on='ride_id',
    )


def drop_id_columns(df):
    return df.drop(['request_id', 'ride_id', 'driver_id'], axis=1)


def haversine_distance(lat1, lon1, lat2, lon2):
    r = 6371  # earth radius in km
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    a = np.sin(delta_phi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(
        delta_lambda / 2) ** 2
    res = r * (2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a)))
    return np.round(res, 2)


def calc_dist(row, from_lat, from_lon, to_lat, to_lon):
    """
    Returns the calculated haversine distance between the geospatial coordinates
    provided in the input arguments
    """
    return haversine_distance(row[from_lat], row[from_lon], row[to_lat],
                              row[to_lon])


def add_distances(df):
    """
    adds three columns of distances in km to the original dataframe and returns
    that dataframe
    """
    df['driver_origin_distance'] = (df
                                    .apply(lambda row: calc_dist(row,
                                                                 "driver_lat",
                                                                 "driver_lon",
                                                                 "origin_lat",
                                                                 "origin_lon"
                                                                 ),
                                           axis=1
                                           )
                                    )

    df['origin_destination_distance'] = (df
                                         .apply(lambda row: calc_dist(row,
                                                                      "origin_lat",
                                                                      "origin_lon",
                                                                      "destination_lat",
                                                                      "destination_lon"
                                                                      ),
                                                axis=1
                                                )
                                         )

    df['driver_destination_distance'] = (df
                                         .apply(lambda row: calc_dist(row,
                                                                      "driver_lat",
                                                                      "driver_lon",
                                                                      "destination_lat",
                                                                      "destination_lon"
                                                                      ),
                                                axis=1
                                                )
                                         )
    return df


def add_timediff(df):
    """
    adds time difference between created and logged
    """

    df['created_logged_time_diff'] = (
        df.apply(lambda row: row.logged_at - row.created_at,
                 axis=1
                 ))

    return df


def create_was_in_ride(row):
    if row.new_state == "ended_ride":
        return True
    elif row.new_state != row.new_state:
        return np.NaN
    else:
        return False


def create_was_connected(row):
    if row.new_state in ('disconnected', 'ended_ride', 'began_ride'):
        return True
    elif row.new_state != row.new_state:
        return np.NaN
    else:
        return False


def get_status_from_id_and_time(row, dr_df, status='was_in_ride'):
    """
    Scans through the merged booking request + ride requests dataframe rowwise.
    For every driver_id and logged_at timestamp value, it looks up the dataframe
    dr derived from the drivers.log table and finds the current state in which
    the driver was in the moment the request to the driver was made.

    status: {"new_state", "previous_state", "was_in_ride", "was_connected"}
    """

    driver_id = row.driver_id
    logged_at = row.logged_at

    # building a temporary df for the specific driver_id
    temp_df = dr_df[dr_df.driver_id == driver_id]

    # need to look up the temp_df for two cases:
    # case1: time of riderequest falls between two log events
    # case2: time of riderequest before first logged event (no data before)
    result_row = temp_df[((temp_df.prev_logged_at_dr < logged_at) & (
            temp_df.logged_at_dr > logged_at)) | ((
                                                          temp_df.prev_logged_at_dr != temp_df.prev_logged_at_dr) & (
                                                          temp_df.logged_at_dr > logged_at))]

    # return value of "search result"
    if len(result_row) == 1:
        return result_row[status].item()

    # but return NaN when search result empty
    else:
        return np.NaN


def get_no_of_past_rides_from_id_and_time(row, dr_):
    """
    Scans through the merged booking request + ride requests dataframe rowwise.
    For every driver_id and logged_at timestamp value, it looks up the dataframe
    dr derived from the drivers.log table and determines how many rides have
     been completed up to that time. Then returns that number.
    """

    driver_id = row.driver_id
    logged_at = row.logged_at

    # building a temporary df for the specific driver_id
    temp_df = dr_[(dr_.driver_id == driver_id)]

    # case: driver only connected and disconnected -> no rides
    if len(temp_df) == 2:
        return 0

    temp_df = temp_df[temp_df.new_state == "began_ride"]

    # add "rides done" column to temp_df
    temp_df['rides_done'] = range(len(temp_df))
    for i in range(len(temp_df)):
        if i == 0 and logged_at < temp_df.iloc[i]['logged_at']:
            # print("if")
            return temp_df.iloc[i]['rides_done']
        elif logged_at > temp_df.iloc[i - 1]['logged_at'] and logged_at < \
                temp_df.iloc[i]['logged_at']:
            # print("elif")
            return temp_df.iloc[i]['rides_done']
        else:
            # print("else")
            return temp_df.iloc[-1]['rides_done']


def add_no_of_past_rides(df, dr):
    print(df.columns)
    df['no_of_accumulated_rides'] = (
        df.apply(lambda row: get_no_of_past_rides_from_id_and_time(row, dr),
                 axis=1))
    return df
