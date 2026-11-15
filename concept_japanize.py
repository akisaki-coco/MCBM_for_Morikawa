import pandas as pd
import os
from openai import OpenAI

def concept_extract(csv_path):
    df = pd.read_csv(csv_path)

    # concept_name列だけを抜きだす
    concept_names = df['concept_name'].dropna().unique()
    return concept_names

def concept_japanize(concept_names):
    # OpenAI API互換のollamaサーバを使う
    client = OpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama"
    )

    japanized_concepts = []
    for concept in concept_names:
        prompt = f"次の英語の概念を日本語に翻訳してください: '{concept}'"
        response = client.chat.Completions.create(
            messages=[
                {"role": "system", "content": "You are a translation assistant that translates English concepts into Japanese."},
                {"role": "user", "content": prompt}
            ],
            model="qwen3.5:27b",
            max_tokens=100,
            tempareture=0,
            extra_body={"think": False}
        )
        japanized_concept = response.choices[0].message.content
        japanized_concepts.append(japanized_concept)

    return japanized_concepts

def save_japanized_concepts(japanized_concepts, output_path):
    # DataFrameに変換してCSVとして保存
    df = pd.DataFrame(japanized_concepts, columns=['japanized_concept'])
    df.to_csv(output_path, index=False)
    print(f"Saved japanized concepts to {output_path}.")

def main():
    # CSVファイルのパスを指定
    csv_path = "outputs/concepts/mydata/concepts_mydata_ollama_qwen3_5_27b_20261115_075936/concept_names.csv"
    output_path = "outputs/concepts/mydata/concepts_mydata_ollama_qwen3_5_27b_20261115_075936/concept_names_japanize.csv"

    # 概念名を抽出
    concept_names = concept_extract(csv_path)

    # 概念名を日本語化
    japanized_concepts = concept_japanize(concept_names)

    # 日本語化された概念名を保存
    save_japanized_concepts(japanized_concepts, output_path)

if __name__ == "__main__":
    main()