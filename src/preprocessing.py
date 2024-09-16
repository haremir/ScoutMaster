import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(df):
    df.fillna(0, inplace=True)
    df['Height'] = df['Height'].apply(lambda x: int(x.replace(' CM', '')) if isinstance(x, str) else 0)
    df['Weight'] = df['Weight'].apply(lambda x: int(x.replace(' KG', '')) if isinstance(x, str) else 0)
    le = LabelEncoder()
    df['Foot'] = le.fit_transform(df['Foot'])
    return df

if __name__ == "__main__":
    data_path = r'C:\Users\emirh\Desktop\DOSYALAR\veri_bilimi\scoutmaster\data\raw_data\train.csv'
    df = load_data(data_path)
    df_processed = preprocess_data(df)
    df_processed.to_csv(r'C:\Users\emirh\Desktop\DOSYALAR\veri_bilimi\scoutmaster\data/processed_data/processed_train.csv', index=False)
    print("Veri başarıyla işlendi ve kaydedildi.")
