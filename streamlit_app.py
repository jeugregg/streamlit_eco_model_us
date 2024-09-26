"""
US Economics model prediction : FEDFUNDS prediction
Streamlit App
TODO : training button, ratio_test  modification
"""
from urllib.error import URLError
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(context='paper', style='whitegrid')
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# definitions
path_unemployment = "data/UNRATE.csv"
path_rate = "data/FEDFUNDS.csv"
path_inflation = "data/CPIAUCSL_08_2024.csv"
path_spx = "data/SP500_history.csv"
ratio_test = 0.017
list_class = ['-0.50','-0.25','0','0.25','0.5']

MODE_TEST = True

@st.cache_data
def get_csv_data(my_path):
    """Reads the CSV file and returns the dataframe
    + rename and format dates
    """
    df_csv = pd.read_csv(my_path)
    if "DATE" in df_csv.columns:
        df_csv["DATE"] = pd.to_datetime(df_csv["DATE"])
    if "Date" in df_csv.columns:
        df_csv.rename(columns={"Date": "DATE"}, inplace=True)
        try:
            df_csv["DATE"] = pd.to_datetime(df_csv["DATE"], format="%b %d, %Y").dt.strftime("%Y-%m-%d")
            df_csv["DATE"] = pd.to_datetime(df_csv["DATE"])
            df_csv.sort_values(by="DATE", inplace=True)
        except:
            print(f"No date format found in column 'date' for {my_path}")
    return df_csv

@st.cache_data
def add_shifted_columns(df, data_columns):
    """
    Adds shifted columns to the dataframe based on the specified data columns.
    
    Parameters:
        df (pd.DataFrame): The original DataFrame.
        data_columns (list): List of column names for which to create shifted columns.
        
    Returns:
        pd.DataFrame: Updated DataFrame with additional shifted columns.
    """
    shift_count = 3
    for col in data_columns:
        for i in range(1, shift_count + 1):
            df[f"{col}-{i}"] = df[col].shift(i)
    
    return df

@st.cache_data
def add_diff_columns(df, data_columns):
    for col in data_columns:
        shifted_cols = [f"{col}-{i}" for i in range(1, 4)]
        df[f"{col}-1_diff"] = df[col] - df[shifted_cols[0]]
        df[f"{col}-2_diff"] = df[shifted_cols[0]] - df[shifted_cols[1]]
        df[f"{col}-3_diff"] = df[shifted_cols[1]] - df[shifted_cols[2]]
    return df
    
# main
st.title("US Economics model prediction")
st.header("FEDFUNDS prediction", divider=True)

# first part of the app :  csv
try:
    df_spx = get_csv_data(path_spx)
    df_spx.rename(columns={"Value": "Close"}, inplace=True)
    df_spx["Close"] = df_spx["Close"].str.replace(",", "").astype(float)
    df_spx["Close-12"] = df_spx["Close"].shift(12)
    df_spx["SPX_diff"] = (df_spx["Close"] - df_spx["Close-12"]) / df_spx["Close-12"]

    df_unemployment = get_csv_data(path_unemployment)
    df_unemployment.rename(columns={"Total": "UNRATE"}, inplace=True)

    df_rate = get_csv_data(path_rate)

    df_inflation = get_csv_data(path_inflation)
    df_inflation["CPI-12"]= df_inflation["CPIAUCSL"].shift(12)
    df_inflation["Inflation"] = 100 * (df_inflation["CPIAUCSL"] - df_inflation["CPI-12"]) / df_inflation["CPI-12"]

    df = df_unemployment.merge(df_inflation.filter(["DATE", "Inflation"]), on="DATE", how="left")
    df["num_month"] = df["DATE"].dt.month

    # merges
    df = df.merge(df_rate, on="DATE", how="left")
    df = df.merge(df_spx.filter(["DATE","SPX_diff"]), how="left", on="DATE")

    # calculate next month diff in unemployment rate 
    df["FEDFUNDS_diff"] = np.diff(df["FEDFUNDS"], append=np.nan)
    # calculate class for fedfunds to predict : targets
    ff_lower_25 = (df["FEDFUNDS_diff"] < -0.05) & (df["FEDFUNDS_diff"] >= -0.25)
    ff_lower_50 = df["FEDFUNDS_diff"] < -0.25 
    ff_higher_25 = (df["FEDFUNDS_diff"] > 0.05) & (df["FEDFUNDS_diff"] <= 0.25)
    ff_higher_50 = df["FEDFUNDS_diff"] > 0.25
    ff_stable = df["FEDFUNDS_diff"].abs() <= 0.05
    # add ur to df
    df["ff_lower_25"] = ff_lower_25
    df["ff_lower_50"] = ff_lower_50
    df["ff_higher_25"] = ff_higher_25
    df["ff_higher_50"] = ff_higher_50
    df["ff_stable"] = ff_stable

    # Assuming you have a DataFrame named `df` and the relevant columns are already loaded.
    data_columns = ["UNRATE", "Inflation", "FEDFUNDS", "SPX_diff"]

    # Apply the function to add shifted columns
    df = add_shifted_columns(df, data_columns)

    # Now `df` contains the new shifted columns: UNRATE-1, UNRATE-2, UNRATE-3,
    # Inflation-1, Inflation-2, Inflation-3, fedfunds-1, fedfunds-2, fedfunds-3,

    # Call the function to add diff columns
    df = add_diff_columns(df, data_columns)

    # inputs for prediction
    df_for_pred = df.iloc[-1:]

    # prepare df
    df.dropna(inplace=True)

    # targets classes
    df["class"] = 0*df["ff_lower_50"] + 1*df["ff_lower_25"] + 2*df["ff_stable"] + 3*df["ff_higher_25"] + 4*df["ff_higher_50"]

    # df_y : just targets
    df_y = df.filter(['FEDFUNDS_diff',
     'ff_lower_50', 'ff_lower_25', 'ff_stable',"ff_higher_25", "ff_higher_50" ,"class"])
    
    # class names
    list_targets = ['ff_lower_50', 'ff_lower_25', 'ff_stable',"ff_higher_25", "ff_higher_50"]

    # split TRAIN/TEST
    nb_test = int(df.shape[0] * (1 - ratio_test))
    df_train = df.iloc[:nb_test].copy()
    df_test = df.iloc[nb_test:].copy()
    df_train["TRAIN"] = 1
    df_test["TRAIN"] = 0
    df = pd.concat([df_train, df_test], axis=0)

    # Select features
    list_feat = [
        'UNRATE', 'Inflation', 'FEDFUNDS', 'SPX_diff', 'num_month',
        'UNRATE-1_diff', 'UNRATE-2_diff', 'UNRATE-3_diff',
        'Inflation-1_diff', 'Inflation-2_diff', 'Inflation-3_diff',
        'FEDFUNDS-1_diff', 'FEDFUNDS-2_diff', 'FEDFUNDS-3_diff',
        'SPX_diff-1', 'SPX_diff-2', 'SPX_diff-3']

    # Disp df
    st.subheader(f"TRAIN Inputs Data: {df_train.shape[0]} rows", divider=True)
    st.dataframe(df_train[["DATE"] + list_feat])
    st.subheader(f"TEST Inputs Data: {df_test.shape[0]} rows", divider=True)
    st.dataframe(df_test[["DATE"] + list_feat])

    # Check Out of range
    df_min = df.filter(list_feat).groupby(df["TRAIN"]).min().transpose()
    df_min["feat"] = df_min.index
    df_min["out_min"] = df_min[0] < df_min[1]
    df_min["pc_out_min"] = df_min["feat"].apply(
        lambda x: 100*sum(df[df["TRAIN"] == 0 ][x] <  df[df["TRAIN"] == 1][x].min()) / df[df["TRAIN"] == 0 ].shape[0]
        )
    
    df_max = df.filter(list_feat).groupby(df["TRAIN"]).max().transpose()
    df_max["feat"] = df_max.index
    df_max["out_max"] = df_max[0] > df_max[1]
    df_max["pc_out_max"] = df_max["feat"].apply(
        lambda x: 100*sum(df[df["TRAIN"] == 0 ][x] >  df[df["TRAIN"] == 1][x].max()) / df[df["TRAIN"] == 0 ].shape[0]
        )
    
    df_range = df_min.copy()
    df_range.drop(columns=["feat", "out_min"], inplace=True)
    df_range.rename_axis(columns=None, inplace=True)
    df_range.rename_axis(index="Features", inplace=True)
    df_range.rename(columns={0: "min TEST", 1: "min TRAIN"}, inplace=True)
    df_range_max = df_max.copy()
    df_range_max.drop(columns=["feat", "out_max"], inplace=True)
    df_range_max.rename_axis(columns=None, inplace=True)
    df_range_max.rename_axis(index="Features", inplace=True)
    df_range_max.rename(columns={0: "max TEST", 1: "max TRAIN"}, inplace=True)
    df_range = df_range.join(df_range_max)
    try:
        assert df_min["pc_out_min"].max() < 5, "Some TEST features are not in Train !"
        assert df_max["pc_out_max"].max() < 5 , "Some TEST features are not in Train !"
        range_ok = True
    except AssertionError as e:
        st.error(e)
        range_ok = False
    
    
    st.subheader(
        f"Check Features Out of range on TRAIN/TEST split: {"OK" if range_ok else "NOK"}",
        divider=True,
    )
    st.dataframe(
        df_range.style.highlight_between(
            subset=["pc_out_min", "pc_out_max"],
            right=0.01,
            color="green",
        ).highlight_between(
            subset=["pc_out_min", "pc_out_max"],
            left=0.01,
        ),
        height=37*df_range.shape[0],
    )

    # Check repart targets
    st.subheader(
        "Distribution Targets on TRAIN/TEST split:",
        divider=True,
    )
    df_class = df.copy()
    
    df_class['classname'] = df_class['class'].apply(lambda x : list_class[int(x)])
    '''def apply_class(class_num):
        """
        change num 2 string    
        """
        if class_num == 0:
            return "-0.50"
        elif class_num == 1:
            return "-0.25"
        elif class_num == 2:
            return "0"
        elif class_num == 3:
            return "0.25"
        elif class_num == 4:
            return "0.5"
        else:
            return None'''
        
    #df_class["classname"] = df_class["class"].apply(apply_class)

    # to keep order in graph convert in categorical
    df_class["classname"] = pd.Categorical(df_class['classname'], list_class)
    g = sns.displot(
        df_class,
        x="classname",
        hue="TRAIN",
        stat="probability",
        common_norm=False,
        discrete=True,
    )
    plt.gca().set_title("Target Distribution : FEDFUNDS variation on TRAIN/TEST split")
    fig = plt.gcf()
    st.pyplot(fig)
    # Scale features

    # feat on train
    xtrain = df[df["TRAIN"] == 1][list_feat].values
    # feat on test
    xtest = df[df["TRAIN"] == 0][list_feat].values
    # target on train
    ytrain = df[df["TRAIN"] == 1]["class"].values
    # target on test
    ytest = df[df["TRAIN"] == 0]["class"].values

    scaler = StandardScaler()

    X = scaler.fit_transform(xtrain)
    X_test = scaler.transform(xtest)

    # for last pred : (to predict next month value)
    x_for_pred = df_for_pred.filter(list_feat).iloc[-1].values.reshape(1, -1)
    X_for_pred = scaler.transform(x_for_pred)

    # prepare Targets
    Y = ytrain.reshape(-1, 1)
    Y_test = ytest.reshape(-1, 1)
    # Y_for_pred : unknown !

    # Correlation
    st.subheader(
        "Correlation Features / Targets on TRAIN split:",
        divider=True,
    )
    def plot_corr(corr_matrix, title='Corrélation with 2 variables', aspect=1, size=25):
        """
        Plot heatmap of correlations of all inputs between them
        """
        fig_size = (size, size*len(corr_matrix.index)/len(corr_matrix.columns))
        fig, ax = plt.subplots(figsize=fig_size)
        im = ax.matshow(corr_matrix, aspect=aspect)
        plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=90)
        plt.yticks(range(len(corr_matrix.index)), corr_matrix.index)
        plt.title(title)
        fig.colorbar(im, aspect=1/aspect, orientation='vertical', location="right")
        return fig
    
    nb_plot = len(list_feat)
    # for only one target : class
    list_col_targets = [ f"target_{n_t}" for n_t in range(len(["class"]))]
    list_col_corr = list_feat[:nb_plot] + list_col_targets
    df_for_corr = pd.DataFrame(np.hstack((X, Y)), columns=list_col_corr)
    corr_matrix = df_for_corr.corr()
    # corr mat only on target
    corr_matrix_targets = corr_matrix.copy().loc[list_feat, list_col_targets]
    fig = plot_corr(corr_matrix_targets, aspect=0.1, size=5)
    st.pyplot(fig)


    # Train
    st.header("Model training", divider=True)
    st.subheader("Decision Tree Classifier", divider=True)
    def choose_target(df_y, target, nb_test):
        print("\nTarget : ", target)
        arr_target = df_y[target].values
        ytrain = arr_target[:nb_test]
        ytest = arr_target[nb_test:]
        Y = ytrain.reshape(-1, 1)
        Y_test = ytest.reshape(-1, 1)
        return Y, Y_test
    def fit_clf(clf, X, Y, X_test, Y_test, X_for_pred):
        clf.fit(X, Y)
        print("TRAIN score :", clf.score(X, Y))
        print("TEST score :", clf.score(X_test, Y_test))
        print("Next month : ", clf.predict(X_for_pred),
        np.max(clf.predict_proba(X_for_pred)[0]),
        )
        return clf
        
    def multi_target_fit(clf, df_y, nb_test, list_targets):
        list_clf = []
        for target in list_targets:
            Y, Y_test = choose_target(df_y, target=target, nb_test=nb_test)
            new_clf = fit_clf(clone(clf), X, Y, X_test, Y_test, X_for_pred)
            list_clf.append(new_clf)
        return list_clf
    

    clf = DecisionTreeClassifier(
        random_state=2,
        max_features=6,
        min_samples_leaf=2,
    )
    list_clf = multi_target_fit(clf, df_y, nb_test, ["class"])
    # display results

    st.subheader(f"Results", divider=True)
    col1, col2 = st.columns(2)
    col1.metric(
        label="***TRAIN accuracy***",
        value=list_clf[-1].score(X, Y),
    )
    col2.metric(
        label="***TEST accuracy***",
        value=list_clf[-1].score(X_test, Y_test),
    )
    # Confusion matrix
    st.subheader(f"Confusion matrix", divider=True)
    predictions = list_clf[-1].predict(X)
    cm = confusion_matrix(Y, predictions, labels=list_clf[0].classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=["-0.5", "-0.25", "0", "0.25", "0.5"])
    disp.plot()
    st.pyplot(disp.figure_)

    # Feature importances
    st.subheader(f"Feature importances", divider=True)
    df_imp = pd.DataFrame(
        index=list_feat,
        data=list_clf[-1].feature_importances_,
        columns=["Importance"]).sort_values(
            by="Importance",
            ascending=False,
    )

    st.dataframe(
        df_imp.style.highlight_between(left=1/(df_imp.shape[0])),
        height=37*df_imp.shape[0],
    )

except URLError as e:
    st.error(
        "**This demo requires internet access." + (
        f"** Connection error: {e.reason}"
        )
    )

