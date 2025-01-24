import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sentence_transformers import SentenceTransformer, util

st.set_page_config(layout="wide")

st.image('beyaz bupa logo.png', width = 200)

st.title('KurumMapping')

st.write('\n')
st.write('\n')

st.markdown("""
    <div style="font-size: 18px; font-weight: bold;">
        Maplemenin yapılabılmesi için yüklenen dosya aşağıdaki gibi olmalıdır:
        <ul>
            <li>Dosya CSV UTF-8 formatında olmalıdır.</li>
            <li>Dosya sadece bir tane sütun (A1 den başlayarak).</li>
            <li>Bu sütun kurum isimlerini içermelidir (ilk satır sütun başlığı olmalıdır).</li>
        </ul>
    </div>
""", unsafe_allow_html=True)

st.write('\n')

@st.cache_resource
def load_references():
    return joblib.load('reference_data_new.pkl')

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

def matched_function(row):
    hc_group = hc_groups[row['kurum_adi']]
    return row['kurum_adi'], 100, hc_group


def unmatched_function(row, reference_dict, encoded_references):  
    category = "OTHER"
    for keyword in reference_dict.keys():
        if keyword in row["kurum_adi"]:
            category = keyword
            break

    filtered_references = reference_dict[category]
    filtered_embeddings = encoded_references[category]

    if not filtered_references or not len(filtered_embeddings):
        st.warning(f"No references found for category: {category}, kurum_adi: {row['kurum_adi']}")
        return None, 0, None

    input_embedding = model.encode(row["kurum_adi"], convert_to_tensor=True)

    cosine_scores = util.cos_sim(input_embedding, filtered_embeddings)

    best_idx = cosine_scores.argmax().item()
    best_score = cosine_scores[0, best_idx].item() * 100  
    best_match = filtered_references[best_idx]

    hc_group = hc_groups.get(best_match, None)

    return best_match, best_score, hc_group

def create_input_df(original_input):

    df = (
        original_input
        .assign(
            kurum_adi=lambda x: x[original_input.columns[0]]
                        .str.strip()
            .str.upper()
            .str.replace('(eski Adı Biruni Üniv.sağ.eğitimi Uygulama Ve Araş. Merk)','')
            .str.replace("İ", "I")
            .str.replace("Ö", "O")
            .str.replace("Ü", "U")
            .str.replace("Ç", "C")
            .str.replace("Ş", "S")
            .str.replace("Ğ", "G")
            .str.replace("?", " ")
            .str.replace("!", " ")
            .str.replace("-", " ")
            .str.replace("_", " ")
            .str.replace("  ", " ")
            .str.replace(" N.HAST", " FLORENCE NIGHTINGALE HASTANESI")
            .str.replace("(ACIBADEM POLIKLINIKLERI A.S.)", "") 
            .str.replace("HASTANESI", "HOSPITAL") 
            .str.replace("HASTANE", "HOSPITAL") 
            .str.replace("LIV HASTANESI", "LIV HOSPITAL") 
            .str.replace("DR.", "DOKTOR ") #
            .str.replace("HİZ.TİC.A.Ş", "") 
            .str.replace("HIZ.TIC", "") 
            .str.replace(" HIZ.", "")
            .str.replace(" TIC.", "")
            .str.replace(" AS.", "")
            .str.replace(" SAG ", " SAGLIK ")
            .str.replace(" SAG.", " SAGLIK ")
            .str.replace("ORTOPEDI", "ORT.")
            .str.replace("OZEL ","")
            .str.replace(" OZEL ","")
            .str.replace(" ECZANE ", " ECZANESI ")
            .str.replace("ECZANE ", "ECZANESI ")
            .str.replace("ECZ.", "ECZANESI ") #
            .str.replace(" ECZ.", " ECZANESI") #
            .str.replace(" ECZ. ", " ECZANESI ") #
            .str.replace(" HAS.", " HOSPITAL") #
            .str.replace(" HAS", " HOSPITAL ") #
            .str.replace(" HAST.", " HOSPITAL")
            .str.replace(" HAST. ", " HOSPITAL ")
            .str.replace("HAST. ", "HOSPITAL ")
            .str.replace(" HAS(", " HOSPITAL(")
            .str.replace(" HAS ", " HOSPITAL ") # riskli
            .str.replace(" UNIVERSITE "," UNIVERSITESI ")
            .str.replace(" UNI ", " UNIVERSITESI ")
            .str.replace(" UNI. ", " UNIVERSITESI ")
            .str.replace(" UNV ", " UNIVERSITESI ")
            .str.replace(" UNV. ", " UNIVERSITESI ")
            .str.replace(" UNIV ", " UNIVERSITESI ")
            .str.replace(" UNIV. ", " UNIVERSITESI ")
            .str.replace("UNV.", "UNIVERSITESI ") #
            .str.replace("UNIV.", "UNIVERSITESI ") #
            .str.replace(" FTR ", " FIZIK TEDAVI VE REHABILITASYON ")   
            .str.replace("TC.", "") #   
            .str.replace("IST.", "ISTANBUL ") #   
            .str.replace("EGT VE ART.", "EGITIM VE ARASTIRMA ") #   
            .str.replace("TIP FAK.", " ") #   
            .str.replace("MRK.", "MERKEZI ") #   
            .str.replace("  ", " ")
        )
)
    return df

st.markdown("""
    <label style="font-size: 18px; font-weight: bold;">
        Kurum Adlarını Buraya Yükleyebilirsiniz:
    </label>
""", unsafe_allow_html=True)

input_file = st.file_uploader('', type=['csv'])

if input_file is not None:

    original_input = pd.read_csv(input_file)

    st.write('Yüklemiş Olduğunuz Dosya')
    st.write('\n')
    st.dataframe(original_input)

    with st.spinner("Mapleme Yapılıyor"):

        df = create_input_df(original_input)
        model = load_model()
        hc_groups, reference_embeddings, reference_dict, encoded_references = load_references()
        #hc_groups, reference_embeddings = load_references()

        reference_list = list(hc_groups.keys())

        df[["HEALTHCENTERDESC", "similarity_score", "HC_GROUP"]] = df.apply(
                lambda row: pd.Series(
                    matched_function(row) if row["kurum_adi"] in reference_list else unmatched_function(row, reference_dict, encoded_references)
                ),
                axis=1
            )
        
    st.success("Mapleme Tamamlandı")
        
    final = (
            df
            .rename(columns = {
                'HEALTHCENTERDESC':'MAPLENDIGI_KURUM_ADI',
                'HC_GROUP':'MAPLENDIGI_KURUM_TIPI',
                'similarity_score': 'YAKINLIK_SKORU'})
            .assign(MANUEL_KONTROL = lambda x: np.where(x['YAKINLIK_SKORU'] <80, 'EVET', 'HAYIR'))
            [[
                df.columns[0],
                'kurum_adi',
                'MAPLENDIGI_KURUM_ADI',
                'MAPLENDIGI_KURUM_TIPI',
                'YAKINLIK_SKORU',
                'MANUEL_KONTROL'
            ]]
                
    )

    st.dataframe(final)
