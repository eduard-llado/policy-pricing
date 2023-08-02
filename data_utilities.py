from functools import cache

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


@cache
def get_raw_data():
    print("Loading raw dataset")
    train = pd.read_excel("data/train.xlsx")
    test = pd.read_excel("data/test.xlsx")
    return train, test


def get_clean_data(remove_all_nulls=False, insured_years_to_ordinal=False):
    # Easier to process all rows together to create bins and so on
    train, test = get_raw_data()
    train_size = len(train)
    df = pd.concat([train, test]).copy()

    # Ignore number_of_competitors since it's not on the test set
    # And also ignore the target variable during cleaning
    df = df[
        [
            "driver_birth_date",
            "driver_driving_license_ym",
            "driver_other_vehicles",
            "driver_insured_years",
            "occasional_driver_birth_date",
            "occasional_driver_license_attainment_age",
            "policyholder_age",
            "policyholder_license_attainment_age",
            "vehicle_acquisition_state",
            "vehicle_buy_ym",
            "vehicle_registration_ym",
            "vehicle_engine_power",
            "vehicle_number_of_doors",
            "vehicle_use",
            "driver_claims_last_1_year",
            "driver_claims_from_year_1_to_2",
            "driver_claims_from_year_2_to_3",
            "driver_claims_from_year_3_to_4",
            "driver_claims_from_year_4_to_5",
            "timestamp",
        ]
    ]

    # Nulls here probably correspond to no insured years
    df.loc[df["driver_insured_years"].isnull(), "driver_insured_years"] = "ZERO"

    if insured_years_to_ordinal:
        years_to_integer = {
            "ZERO": 0,
            "ONE": 1,
            "TWO": 2,
            "THREE": 3,
            "FOUR": 4,
            "FIVE": 5,
            "SIX": 6,
            "SEVEN": 7,
            "EIGHT": 8,
            "NINE": 9,
            "TEN": 10,
            "MORE_THAN_TEN": 11,
        }

        df["driver_insured_more_than_ten_years"] = (
            df.loc[:, "driver_insured_years"] == "MORE_THAN_THEN"
        )
        df.loc[:, "driver_insured_years"] = df.loc[:, "driver_insured_years"].apply(
            lambda x: years_to_integer[x]
        )

    if remove_all_nulls:  # Linear models can't handle null values
        # Nulls in these numeric variables are relevant and it doesn't make sense
        # to impute them so I decided to make them categorical
        to_categorical = [
            "occasional_driver_birth_date",
            "occasional_driver_license_attainment_age",
            "policyholder_age",
            "policyholder_license_attainment_age",
        ]
        for col in to_categorical:
            df[col] = pd.cut(df[col], bins=8, labels=False).astype(str)

    # Fields in year-month format to datetime
    ym_format_cols = [
        "driver_driving_license_ym",
        "vehicle_buy_ym",
        "vehicle_registration_ym",
    ]
    for col in ym_format_cols:
        df[col] = pd.to_datetime(df[col])

    # Datetime -> int, categorical -> one-hot encode
    for col in df:
        dtype = str(df[col].dtype)
        if dtype == "object":
            dummies = pd.get_dummies(df[col])
            df[col + "#" + dummies.columns] = dummies
            df = df.drop([col], axis=1)
        elif dtype == "datetime64[ns]":
            df[col] = df[col].astype(int)

    X_train, X_test = df.iloc[:train_size:, :], df.iloc[train_size:, :]
    y_train = train["competitor_lowest_price"]
    return X_train, y_train, X_test


def get_split(remove_all_nulls=False):
    X, y, _ = get_clean_data(remove_all_nulls=remove_all_nulls)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=57
    )
    return X_train, y_train, X_test, y_test
