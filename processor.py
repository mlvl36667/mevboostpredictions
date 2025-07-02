import pandas as pd
from tqdm import tqdm
import os
import json
from scipy.stats import linregress
def create_df_from_file(file_path):
 data = []
 current_block = ""
 with open(file_path, "r") as file:
     for line in file:
         if line.strip() == "}":
             current_block += line.strip()
             try:
                 data.append(json.loads(current_block))
             except json.JSONDecodeError as e:
                 print(f"Hiba történt a blokk dekódolásakor: {e}")
             current_block = ""
         else:
             current_block += line.strip()
 df = pd.DataFrame(data)
 base_timestamp_ms = int(df['timestamp_ms'].min())
 wei_to_eth = 10**18
 df['value'] = pd.to_numeric(df['value'], errors='coerce')
 maximal_bid = df['value'].max() / wei_to_eth
 df['relative_timestamp'] = (df['timestamp_ms'].astype(int) - base_timestamp_ms) / 1000
 df_sorted = df.sort_values(by=["timestamp_ms"]).reset_index(drop=True)
 df_sorted["second"] = df_sorted["relative_timestamp"].astype(int)
 counts_per_second = df_sorted.groupby("second").size().reset_index(name="bid_count")
 unique_builder_pubkeys_per_second = df_sorted.groupby("second")["builder_pubkey"].nunique().reset_index(name="unique_builder_pubkeys")
 df_sorted["value"] = df_sorted["value"].astype(float)
 df_sorted["value_in_eth"] = df_sorted["value"] / wei_to_eth
 statistics_per_second = df_sorted.groupby("second")["value_in_eth"].agg(
     average_value="mean",
     std_deviation="std",
     max_value="max",
     min_value="min"
 ).reset_index()
 statistics_per_second['std_deviation'] = statistics_per_second['std_deviation'].fillna(0)
 slopes = []
 intercepts = []  # Létrehozzuk az intercepts listát
 for second, group in df_sorted.groupby("second"):
     if len(group) > 1:  # Csak akkor illesztünk, ha több pont van
         x = group["relative_timestamp"]
         y = group["value_in_eth"]
 
         # Ellenőrizzük, hogy az x értékek nem azonosak-e
         if len(set(x)) > 1:
             slope, intercept, _, _, _ = linregress(x, y)
             slopes.append(slope)
             intercepts.append(intercept)
         else:
             slopes.append(0)  # Ha az x értékek azonosak, akkor a meredekség 0
             intercepts.append(min(y))  # Az intercept legyen a minimum érték
     else:
         slopes.append(0)  # Ha csak egy pont van, meredekség 0
         intercepts.append(min(group["value_in_eth"]))  # Az intercept legyen a minimum érték
 statistics_per_second["slope"] = slopes
 statistics_per_second["intercept"] = intercepts
# print(statistics_per_second)
 
 merged_df = pd.merge(unique_builder_pubkeys_per_second, statistics_per_second, on="second", how="left")
 merged_df2 = pd.merge(counts_per_second, merged_df, on="second", how="left")
 merged_df2_second_0 = merged_df2[merged_df2['second'] == 0]
 merged_df2_second_1 = merged_df2[merged_df2['second'] == 1]
 merged_df2_second_2 = merged_df2[merged_df2['second'] == 2]
 merged_df2_second_3 = merged_df2[merged_df2['second'] == 3]
 merged_df2_second_4 = merged_df2[merged_df2['second'] == 4]
 merged_df2_second_5 = merged_df2[merged_df2['second'] == 5]
 merged_df2_second_6 = merged_df2[merged_df2['second'] == 6]
 merged_df2_second_7 = merged_df2[merged_df2['second'] == 7]
 merged_df2_second_8 = merged_df2[merged_df2['second'] == 8]
 merged_df2_second_9 = merged_df2[merged_df2['second'] == 9]
 merged_df2_second_10 = merged_df2[merged_df2['second'] == 10]
 merged_df2_second_11 = merged_df2[merged_df2['second'] == 11]
 merged_df2_second_12 = merged_df2[merged_df2['second'] == 12]
 merged_df2_second_0 =  merged_df2_second_0.rename(columns=lambda x: f"{x}_0")
 merged_df2_second_1 =  merged_df2_second_1.rename(columns=lambda x: f"{x}_1")
 merged_df2_second_2 =  merged_df2_second_2.rename(columns=lambda x: f"{x}_2")
 merged_df2_second_3 =  merged_df2_second_3.rename(columns=lambda x: f"{x}_3")
 merged_df2_second_4 =  merged_df2_second_4.rename(columns=lambda x: f"{x}_4")
 merged_df2_second_5 =  merged_df2_second_5.rename(columns=lambda x: f"{x}_5")
 merged_df2_second_6 =  merged_df2_second_6.rename(columns=lambda x: f"{x}_6")
 merged_df2_second_7 =  merged_df2_second_7.rename(columns=lambda x: f"{x}_7")
 merged_df2_second_8 =  merged_df2_second_8.rename(columns=lambda x: f"{x}_8")
 merged_df2_second_9 =  merged_df2_second_9.rename(columns=lambda x: f"{x}_9")
 merged_df2_second_10 = merged_df2_second_10.rename(columns=lambda x: f"{x}_10")
 merged_df2_second_11 = merged_df2_second_11.rename(columns=lambda x: f"{x}_11")
 merged_df2_second_12 = merged_df2_second_12.rename(columns=lambda x: f"{x}_12")
 
 result_df = pd.concat([
     merged_df2_second_0.reset_index(drop=True),
     merged_df2_second_1.reset_index(drop=True),
     merged_df2_second_2.reset_index(drop=True),
     merged_df2_second_3.reset_index(drop=True),
     merged_df2_second_4.reset_index(drop=True),
     merged_df2_second_5.reset_index(drop=True),
     merged_df2_second_6.reset_index(drop=True),
     merged_df2_second_7.reset_index(drop=True),
     merged_df2_second_8.reset_index(drop=True),
     merged_df2_second_9.reset_index(drop=True),
     merged_df2_second_10.reset_index(drop=True),
     merged_df2_second_11.reset_index(drop=True),
     merged_df2_second_12.reset_index(drop=True)
 ], axis=1)
 result_df = result_df.drop(columns=[col for col in result_df.columns if 'second' in col])
 result_df['auction_length'] = 0
 for i in range(13):
     col_name = f'bid_count_{i}'
     result_df.loc[result_df[col_name] > 0, 'auction_length'] = i + 1
 result_df['block_number']=file_path
 result_df['maximal_bid']=maximal_bid
 # Numerikus értékek kategorizálása egész számként

 result_df['block_number'] = result_df['block_number'].str.replace("block_", "").str.replace(".json", "")
 result_df = pd.concat([result_df['block_number'], result_df.drop(columns=['block_number'])], axis=1)
 return result_df
input_folder = "/home/e/pbs_data/"  # A mappa, ahol a fájlok találhatóak
output_file = "output/12s_auctions.csv"  # Az összesített CSV fájl neve
all_files = [f for f in os.listdir(input_folder) if f.endswith(".json")]
all_results = []
files_to_process = all_files[:150000]
combined_df = pd.DataFrame()
for file_index, file_name in enumerate(tqdm(files_to_process, desc="Processing files")):
    file_path = os.path.join(input_folder, file_name)
    if os.path.getsize(file_path) > 0:
        results_df = create_df_from_file(file_path)
        all_results.append(results_df)
        if not results_df.empty:  # Csak nem üres DataFrame-eket ad hozzá
         combined_df = pd.concat([combined_df, results_df], ignore_index=True)
combined_df = pd.concat(all_results, ignore_index=True)
combined_df['block_number'] = combined_df['block_number'].str.extract(r'(\d+)$')
combined_df['high_mev_block'] = combined_df["maximal_bid"] > 1
combined_df['high_mev_block'] = combined_df['high_mev_block'].astype(int)
#combined_df = combined_df[combined_df["maximal_bid"] < 1]
combined_df = combined_df.round(5)

combined_df['bid_category'] = pd.cut(
    combined_df['maximal_bid'],
    bins=20,
    labels=range(1, 21)  # Egész számokat használunk a címkékre
)

combined_df['bid_category'] = combined_df['bid_category'].astype(int)  # Típus konvertálás egész számra

combined_df.to_csv(output_file, index=False)
print(f"Mentés kész: {output_file}")
