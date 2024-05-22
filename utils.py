def concordance_correlation_coefficient(y_true, y_pred): # calculate CCC
    import pandas as pd
    import numpy as np
    """Concordance correlation coefficient."""
    # Raw data
    dct = {
        'y_true': y_true,
        'y_pred': y_pred
    }
    df = pd.DataFrame(dct)
    # Remove NaNs
    df = df.dropna()
    # Pearson product-moment correlation coefficients
    y_true = df['y_true']
    y_pred = df['y_pred']
    cor = np.corrcoef(y_true, y_pred)[0][1]
    # Means
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    # Population variances
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    # Population standard deviations
    sd_true = np.std(y_true)
    sd_pred = np.std(y_pred)
    # Calculate CCC
    numerator = 2 * cor * sd_true * sd_pred
    denominator = var_true + var_pred + (mean_true - mean_pred) ** 2

    return numerator / denominator

def countPredictionInt(prediction):
    import collections

    # new_prediction = prediction.replace("\n", "").split(" ")
    # total_count = len(new_prediction)
    total_count = len(prediction)

    frequency = collections.Counter(prediction)
    frequency_dict = dict(frequency)

    very_negative = negative = neutral = positive = very_positive = 0

    for key, value in frequency_dict.items():
        # print("key:", key, "value:", value)
        if key == "very_negative":
            very_negative = value
        elif key == "negative":
            negative = value
        elif key == "neutral":
            neutral = value
        elif key == "positive":
            positive = value
        elif key == "very_positive":
            very_positive = value
    return very_negative, negative, neutral, positive, very_positive, total_count

def read_csv_to_df(file):
    import pandas as pd
    df = pd.read_csv(file, encoding='utf-8')
    return df

