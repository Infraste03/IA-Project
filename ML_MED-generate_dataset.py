import pandas as pd
from sklearn import preprocessing
import unidecode
from collections import defaultdict
import math
import random
import numpy as np
from tqdm import tqdm
import os
import datetime

import warnings

warnings.filterwarnings('ignore')

DATASET_GESTIONALE = False

comorbidita_dict = {
    "OBIESITA": "OBESITA",
    "OBIESITÀ": "OBESITA",
    "OBIESITA'": "OBESITA",
    "OBESITA": "OBESITA",
    "OBESITÀ'": "OBESITA",
    "IPERCOLESTREOLEMIA": "IPERCOLESTEROLEMIA",
    "OBESITA'": "OBESITÀ",
    "CARDIOPATRIA ISCHIMICA CRONICA": "CARDIOPATIA ISCHEMICA CRONICA",
    "IPERTENSIONE ARTERIOSO": "IPERTENSIONE ARTERIOSA",
    "IPOTIERODISMO": "IPOTIROIDISMO",
    "IPERPLASSIA SURENALICA": "IPERPLASIA SURRENALICA",
    "ADENOMA PROSTATICA": "ADENOMA PROSTATICO",
    "S. DEPRESSIVA": "SINDROME DEPRESSIVA",
    "ENDOMETROSI": "ENDOMETRIOSI",
    "SINDROME ANSIOSO-DEPRESSIVA": "SINDROME DEPRESSIVA",
    "PROGRESO TVP": "PREGRESSO TVP",
    "OVAIA POLICISTICA": "OVAIO POLICISTICO",
    "ADENOMA MAMMARIA": "ADENOMA MAMMARIO",
    "LINFOMA GASTRICA": "LINFOMA GASTRICO",
    "STENT ANEURISMO": "STENT ANEURISMA",
    "DISPLEDIMIA": "DISLIPIDEMIA",
    "PROGRESSO INFARTO MIOCARDICO": "PREGRESSO INFARTO MIOCARDIO",
    "BPCO ASMATIFORMR": "BPCO ASMATIFORME",
    "PRORESSO TIA": "PREGRESSO TIA",
    "PROGRESSO TIA": "PREGRESSO TIA",
    "CARDIOPATIA ISCHIMICA": "CARDIOPATIA ISCHEMICA",
    "DISLIPEDEMIA": "DISLIPIDEMIA",
    "DISPLIDEMIA": "DISLIPIDEMIA",
    "DISLIPOÌIDEMIA": "DISLIPIDEMIA",
    "PROGRESSO INFARTO MIOCARDIO": "PREGRESSO INFARTO MIOCARDIO",
    "ANEMIA MEDITERRAN": "ANEMIA MEDITERRANEA",
    "SECONDO": "II",
    "PREGRESSO INFARTO DEL MIOCARDIO": "PREGRESSO INFARTO MIOCARDIO",
    "GLAUCOM": "GLAUCOMA",
    "DEPRESSIONE": "SINDROME DEPRESSIVA",
    "IPERTENSIONE ARTERIOAS": "IPERTENSIONE ARTERIOSA",
    "NOSULO TIROIDEO": "NODULO TIROIDEO",
    "S.DEPRESSIVA": "SINDROME DEPRESSIVA",
    "S. ANSIOSO-DEPRESSIVA": "SINDROME DEPRESSIVA",
    "IPERTENSIONE AARTERIOSA": "IPERTENSIONE ARTERIOSA",
    "DETERIORMNTO COGNITIVO": "DETERIORAMENTO COGNITIVO",
    "IPERTENSIONE RTERIOSA": "IPERTENSIONE ARTERIOSA",
    "GLAUCOMAA": "GLAUCOMA",
    "GLAUCOME": "GLAUCOMA",
    "IPOTIROISIMO": "IPOTIROIDISMO",
    "IPOTIROIDIOSMO": "IPOTIROIDISMO",
    "DISLPIDEMIA": "DISLIPIDEMIA",
    "VARICI ART .INF": "VARICI ARTI INF",
    "TRAID TALASIMICP": "TRAIT TALASSEMICO",
    "DISLIDEMIA": "DISLIPIDEMIA",
    "DISLIPEDIMIA": "DISLIPIDEMIA",
    "ARITMIE": "ARITMIA",
    "DISLIPIEDEMIA": "DISLIPIDEMIA",
    "DISLIPIDEMIA ": "DISLIPIDEMIA",
    "DISLIPIDIMIA": "DISLIPIDEMIA",
    "DISTURDO SCHIZOTIPICO": "DISTURBO SCHIZOTIPICO",
    "FUMO ": "FUMO",
    "INTOLLERANZA GLICIDICA": "INTOLLERANZA GLUCIDICA",
    "IPERTENSIONE ARTERIOSA ": "IPERTENSIONE ARTERIOSA",
    "IPOTIROIDISMO ": "IPOTIROIDISMO",
    "IPOTIRPIDISMO": "IPOTIROIDISMO",
    "S.DEPRISSIVA": "SINDROME DEPRESSIVA",
    "SD ANSIOSA": "SINDROME ANSIOSA",
    "FIBROMIALGIA ": "FIBROMIALGIA",
    "K.POLMONARE": "K POLMONARE",
    "S. ANSIOSA": "SINDROME ANSIOSA",
    "ANSIA": "SINDROME ANSIOSA",
    "CARDIOPATIA DILATATIVA": "CARDIOMIOPATIA DILATATIVA",
    "EPATOPATIA HCV RELATA": "EPATOPATIA HCV CORRELATA",
    "IOTIROIDISMO": "IPOTIROIDISMO",
    "IPEURICEMIA": "IPERURICEMIA",
    "IPOTIROIDSIMO": "IPOTIROIDISMO",
    "OIPOTIROIDISMO": "IPOTIROIDISMO",
    "POLMONITE DA COVID": "POLMONITE COVID",
    "PREGRESSO INFARTO MIOCARDICO": "PREGRESSO INFARTO MIOCARDIO",
    "TRAPIANTO DI RENE": "TRAPIANTO RENALE"
}

terapia_pre_dict = {
    "AMILODIPINA": "AMLODIPINA",
    "ALLUPRONOLOLO": "ALLOPURINOLO",
    "BISOPROLO": "BISOPROLOLO",
    "DELTACORTONE": "DELTACORTENE",
    "DILATERND": "DILATREND",
    "DILATRENT": "DILATREND",
    "EUTIROPX": "EUTIROX",
    "INDURAL": "INDERAL",
    "LASIXX": "LASIX",
    "MICONFENOLATTO": "MICONFENOLATO",
    "MIRTAAPINA": "MIRTAZAPINA",
    "NATCAL": "NATECAL",
    "NEBIVILOLO": "NEBIVOLOLO",
    "OMONIC": "OMNIC",
    "OMREPAZOLO": "OMEPRAZOLO",
    "PANTAPRAZOLO": "PANTOPRAZOLO",
    "QUETAPINA": "QUETIAPINA",
    "SERATOLINA": "SERTRALINA",
    "SSALBUTAMOLO": "SALBUTAMOLO",
    "STATINA": "STATINE",
    "TARDIFER": "TARDYFER",
    "TMSULOSINA": "TAMSULOSINA",
    "URORETIC": "UROREC",
    "VELPTTORO": "VELPHORO",
    "NO": "NO_TERAPIA"
}

lateralit_pre_dict = {
    "0.0": "LATERALITA_NESSUNA",
    "1.0": "LATERALITA_DESTRA",
    "2.0": "LATERALITA_SINISTRA",
    "3.0": "LATERALITA_BILATERALE",

}

mallampati_pre_dict = {
    "1": "MALLAMPATI_1",
    "2": "MALLAMPATI_2",
    "3": "MALLAMPATI_3",
    "4": "MALLAMPATI_4",
    "5": "MALLAMPATI_5",
}

asa_pre_dict = {
    "1": "ASA_1",
    "2": "ASA_2",
    "3": "ASA_3",
    "4": "ASA_4",
    "5": "ASA_5",
}

catetere_vescicale_pre_dict = {
    "NO": "CATETERE_VESCIALE_NO",
    "SI": "CATETERE_VESCIALE_SI",
    "no": "CATETERE_VESCIALE_NO",
}

cvc_pre_dict = {
    "NO": "CVC_NO",
    "SI": "CVC_SI",
    "no": "CVC_NO"
}

reintervento_pre_dict = {
    "NO": "REINTERVENTO_NO",
    "SI": "REINTERVENTO_SI",
    "no": "REINTERVENTO_NO",
}

accesso_chirurgico_predict = {
    "1": "OPEN",
    "2": "LAPAROSCOPIA",
    "3": "ENDOSCOPIA",
    "4": "TORACOSCOPIA",
    "5": "ROBOTICA",
    "6": "PERCUTANEO"
}

diabete_mellito_predict = {
    "0.0": "NO_DIABETE_MELLITO",
    "1.0": "DIABETE_MELLITO_1",
    "2.0": "DIABETE_MELLITO_2",
    "3.0": "DIABETE_MELLITO_3",
    "4.0": "DIABETE_MELLITO_4",
    "5.0": "DIABETE_MELLITO_5",
    "6.0": "DIABETE_MELLITO_6"
}

diagnosis = {
    (1, 139): "Malattie infettive e parassitarie",
    (140, 239): "Tumori",
    (240, 279): "Malattie delle ghiandole endocrine, della nutrizione e del metabolismo, e disturbi immunitari",
    (280, 289): "Malattie del sangue e organi emopoietici",
    (290, 319): "Disturbi Mentali",
    (320, 389): "Malattie del sistema nervoso e degli organi di senso",
    (390, 459): "Malattie del sistema circolatorio",
    (460, 519): "Malattie dell’apparato respiratorio",
    (520, 579): "Malattie dell’apparato digerente",
    (580, 629): "Malattie dell’apparato genitourinario",
    (630, 677): "Complicazioni della gravidanza, del parto e del puerperio",
    (680, 709): "Malattie della pelle e del tessuto sottocutaneo",
    (710, 739): "Malattie del sistema osteomuscolare e del tessuto connettivo",
    (740, 759): "Malformazioni congenite",
    (760, 779): "Alcune condizioni morbose di origine perinatale",
    (780, 799): "Sintomi, segni, e stati morbosi maldefiniti",
    (800, 999): "Traumatismi e avvelenamenti"
}

surgeries = {
    (0, 1): "PROCEDURE ED INTERVENTI NON CLASSIFICATI ALTROVE",
    (1, 5): "INTERVENTI SUL SISTEMA NERVOSO",
    (6, 7): "INTERVENTI SUL SISTEMA ENDOCRINO",
    (18, 20): "INTERVENTI SULL'ORECCHIO",
    (21, 29): "INTERVENTI SU NASO, BOCCA E FARINGE",
    (30, 34): "INTERVENTI SUL SISTEMA RESPIRATORIO",
    (35, 39): "INTERVENTI SUL SISTEMA CARDIOVASCOLARE",
    (40, 41): "INTERVENTI SUL SISTEMA EMATICO E LINFATICO",
    (42, 54): "INTERVENTI SULL’APPARATO DIGERENTE",
    (55, 59): "INTERVENTI SULL’APPARATO URINARIO",
    (60, 64): "INTERVENTI SUGLI ORGANI GENITALI MASCHILI",
    (65, 71): "INTERVENTI SUGLI ORGANI GENITALI FEMMINILI",
    (72, 75): "INTERVENTI OSTETRICI",
    (76, 84): "INTERVENTI SULL’APPARATO MUSCOLOSCHELETRICO",
    (85, 86): "INTERVENTI SUI TEGUMENTI",
    (87, 99): "MISCELLANEA DI PROCEDURE DIAGNOSTICHE E TERAPEUTICHE"
}

coloumn_to_save = ['Codice alfa numerico', 'Età', 'Sesso', 'Peso', 'Altezza', 'BMI', 'ASA', 'Mallampati',
                   'Cormack', 'Catetere vescicale', 'CVC', 'Diagnosi', 'Codice diagnosi', "Diabete Mellito", "Fumo",
                   "OSAS", "Pregressa polmonite(>30 gg)", "BPCO", "Ipertensione arteriosa",
                   "Cardiopatia ischemica cronica",
                   "Pregresso infarto miocardio", "Pregresso SCC", "Aritmie", "Ictus", "Pregresso TIA",
                   "Note_comorbidita", "Antipertensivi",
                   "Broncodilatatori", "Antiaritmici", "Anticoagulanti", "Antiaggreganti", "TIGO", "Insulina",
                   "Note_medicinali",
                   'Intervento', 'Codice intervento', 'Reintervento', 'Lateralità',
                   'Accesso chirurgico', 'Tempo Tot BO Ormaweb', 'Tempo Tot. SO OrmaWeb', 'Tempo Tot. RR',
                   'Numero chirurghi', 'Specializzando chirurgia', 'Numero anestesisti', 'Cambio anestesisti',
                   'Specializzando anestesia', 'Altro_comorbidita', "Altro_terapia"]


def common_data(list1, list2):
    result = False

    # traverse in the 1st list
    for x in list1:

        # traverse in the 2nd list
        for y in list2:

            # if one common
            if x == y:
                result = True
                return result

    return result


def onehotencoding_forMultiLabelRow(dataset, column_name):
    # Per la Comorbidità dobbiamo costruire un hot encoder personalizzato, idem per gl interventi
    output_df = pd.DataFrame()

    for index, row in dataset.iterrows():
        for content in str(row[column_name]).split(","):
            newcontet = content.lstrip().rstrip()
            if newcontet != "" and newcontet:

                newcontet = unidecode.unidecode(newcontet)  # rimozione degli accenti

                if column_name == "Note_medicinali":
                    substitution = terapia_pre_dict
                elif column_name == "Lateralità":
                    substitution = lateralit_pre_dict
                elif column_name == "Accesso chirurgico":
                    substitution = accesso_chirurgico_predict
                elif column_name == "Note_comorbidita":
                    substitution = comorbidita_dict
                elif column_name == "Mallampati":
                    substitution = mallampati_pre_dict
                elif column_name == "ASA":
                    substitution = asa_pre_dict
                elif column_name == "Diabete Mellito":
                    substitution = diabete_mellito_predict
                else:
                    pass

                if 'substitution' in locals():
                    for key, value in substitution.items():  # sostituzione di abbreviazioni ed errori di scrittura
                        #print("newcon", newcontet)
                        if key in newcontet:

                            newcontet = newcontet.replace(key, value)
                            #print("ciccio", newcontet)
                #print(output_df)
                output_df.loc[index, newcontet] = 1
    output_df = output_df.fillna(0)
    return output_df


def onehotencoding_forMultiLabelRow_forSURGERIES(dataset, column_name):
    # Per la Comorbidità dobbiamo costruire un hot encoder personalizzato, idem per gl interventi
    output_df = pd.DataFrame()
    to_remove = []
    for index, row in dataset.iterrows():
        lenght_procedures = len(str(row[column_name]).split(","))
        content = str(row[column_name]).split(",")[0]
        newcontet = content.lstrip().rstrip().replace("V", "")
        if newcontet != "" and newcontet != "nan" and newcontet:
            for key, value in surgeries.items():
                try:
                    if key[0] <= math.floor(float(newcontet)) <= key[1]:
                        to_add = value
                except:
                    # devo contare quante ore e quanti minuti sono passati dal primo gennaio del 1900
                    date_time_obg = datetime.datetime.strptime(newcontet, '%Y-%m-%d %H:%M:%S')
                    date_time_to_make_difference = datetime.datetime(1899, 12, 31, 0, 0, 0)

                    difference = date_time_obg - date_time_to_make_difference

                    new_code = (difference.total_seconds() // 3600) + (difference.total_seconds() - (
                            3600 * math.floor(difference.total_seconds() / 3600))) / (60 * 100)

                    if key[0] <= math.floor(float(newcontet)) <= key[1]:
                        to_add = value
            output_df.loc[index, to_add] = 1
            output_df.loc[index, "PROCEDURE_TOTALI"] = lenght_procedures
        else:
            # to_remove.append(index)
            output_df.loc[index, "NO_INTERVENTO"] = 1
            output_df.loc[index, "PROCEDURE_TOTALI"] = 0
    output_df = output_df.fillna(0)
   # print(list(output_df.index))
    return output_df


BLE_Data = pd.read_csv("BLE_Data/raw_data.csv")
df = pd.read_excel("EHR_Data/BLOC-OP statistica.xlsx")
data = df[df.columns.intersection(coloumn_to_save)]

range2 = range(787, df.shape[0])
# data.drop([0], axis=0, inplace=True)
data.drop(list(range2), axis=0, inplace=True)
if not DATASET_GESTIONALE:
    data.drop([125, 383, 395], inplace=True)

###################################################################################################################

# Puliamo un po' i dati elimando quelli che non hanno le features che ci servono
if not DATASET_GESTIONALE:
    data = data[(data["Peso"].notna()) & (data["Tempo Tot BO Ormaweb"] != 0)]

rooms = ["Sala_Operatoria_1", "Sala_Operatoria_2", "Sala_Operatoria_4"]

###################################################################################################################


#### Uniamo i dati anamnestici con i dati provenienti dall'architettura IOT ###
#### Cerchiamo anche di recuperare i pazienti che non hanno rilevazioni nella recovery room ####
for index, row in data.iterrows():
    ml_med_code = row["Codice alfa numerico"]

    if ml_med_code in list(BLE_Data["identification_code"]):
        current_id = BLE_Data[BLE_Data["identification_code"] == ml_med_code]
        data.loc[index, "feasible"] = list(current_id["feasible"])[0]
        data.loc[index, "BLE_tot_BO_time"] = round(sum(list(current_id["time_duration_minutes"])), 2)

        for room in rooms:
            if room in list(current_id["room"]):
                current_room = current_id.loc[current_id["room"] == room]
                data.loc[index, "BLE_tot_OR_time"] = round(sum(list(current_room["time_duration_minutes"])), 2)
                break
            else:
                data.loc[index, "BLE_tot_OR_time"] = 0

        if "Recovery_Room" in list(current_id["room"]):
            current_rr = current_id.loc[current_id["room"] == "Recovery_Room"]
            data.loc[index, "BLE_tot_RR_time"] = round(sum(list(current_rr["time_duration_minutes"])), 2)
        else:
            if data.loc[index, "Tempo Tot. RR"] <= 10:
                data.loc[index, "feasible"] = True
                data.loc[index, "BLE_tot_RR_time"] = 0
            else:
                data.loc[index, "BLE_tot_RR_time"] = 0

    else:
        data.loc[index, "BLE_tot_BO_time"] = 0
        data.loc[index, "BLE_tot_RR_time"] = 0
        data.loc[index, "BLE_tot_OR_time"] = 0
        data.loc[index, "feasible"] = False

###################################################################################################################
# riempimento dei dati nulli
data["ASA"].fillna(value=2, inplace=True)
data["Mallampati"].fillna(value=1, inplace=True)

lista_to_fill = ["Diabete Mellito", "Fumo", "OSAS", "Pregressa polmonite(>30 gg)", "BPCO", "Ipertensione arteriosa",
                 "Cardiopatia ischemica cronica",
                 "Pregresso infarto miocardio", "Pregresso SCC", "Aritmie", "Ictus", "Pregresso TIA",
                 "Antipertensivi", "Broncodilatatori", "Antiaritmici", "Anticoagulanti", "Antiaggreganti", "TIGO",
                 "Insulina", "Altro_comorbidita", "Altro_terapia",
                 "CVC", "Reintervento", "Catetere vescicale"]

for elm in lista_to_fill:
    data[elm].fillna(value=0, inplace=True)

###################################################################################################################

### costruiamo anche un  dizionario che mappa che associa il codice intervento all'intervento stesso
codes = defaultdict()
for index1, row in data.iterrows():
    if isinstance(row["Codice intervento"], datetime.datetime):
        # devo contare quante ore e quanti minuti sono passati dal primo gennaio del 1900
        key = str(row["Codice intervento"])

        date_time_obg = datetime.datetime.strptime(key, '%Y-%m-%d %H:%M:%S')
        date_time_to_make_difference = datetime.datetime(1899, 12, 31, 0, 0, 0)

        difference = date_time_obg - date_time_to_make_difference

        key = (difference.total_seconds() // 3600) + (
                difference.total_seconds() - (3600 * math.floor(difference.total_seconds() / 3600))) / (60 * 100)

        data.loc[index1, "Codice intervento"] = key
    elif isinstance(row["Codice intervento"], datetime.time):
        key = str(row["Codice intervento"])
        date_time_obg = datetime.datetime.strptime(key, '%H:%M:%S').time()

        key = float(str(date_time_obg.hour) + "." + str(date_time_obg.minute))

        data.loc[index1, "Codice intervento"] = key

###################################################################################################################

OH_encoded_Column = ["ASA", "Mallampati", "Accesso chirurgico", "Lateralità", "Diabete Mellito"]
# "Codice intervento"

for column in OH_encoded_Column:
    df_encoded = onehotencoding_forMultiLabelRow(data, column)
    data = pd.concat([data, df_encoded.set_index(data.index)], axis=1)

#df_encoded_diagnosis = onehotencoding_forMultiLabelRow_fordiagnosis(data, "Codice diagnosi")
#data = pd.concat([data, df_encoded_diagnosis.set_index(data.index)], axis=1)

df_encoded_surgeries = onehotencoding_forMultiLabelRow_forSURGERIES(data, "Codice intervento")
data = pd.concat([data, df_encoded_surgeries.set_index(data.index)], axis=1)
###################################################################################################################

# dobbiamo aggiornare i vari valori del codice intervento perché alcuni sono formattati come dati o orari
# Codice per contare le tipologie di intervento presenti nel database, sarà utile in future per il bilanciamento
# del dataset
# data.drop(data[data["Codice intervento"] == "nan"].index, inplace=True)

''''
count = {}
for cnt, val in enumerate(data["Codice intervento"].apply(lambda x: str(x).split(","))):
    for elm in val:
        element = math.floor(float(str(elm).lstrip().rstrip()))
        if element in count:
            count[element] += 1
        else:
            count[element] = 1

count = dict(sorted(count.items(), key=lambda item: item[1]))
for key in count:
    print(key, ":", count[key])

### costruiamo anche un  dizionario che mappa che associa il codice intervento all'intervento stesso
codes = defaultdict()
for index, row in data.iterrows():
    for index, value in enumerate(str(row["Codice intervento"]).split(",")):
        surgery = row["Intervento"].split(",")
        if index < len(surgery):
            key = value.rstrip().lstrip()
            if key not in codes:
                codes[key] = surgery[index].rstrip().lstrip()

'''
count = {}
for value in surgeries.values():
    if value in data:
        count[value] = data[value].value_counts().to_dict()[1]

for key, value in count.items():
    print(key, ":", value)
###################################################################################################################

###################################################################################################################
data.pop("Intervento")
data.pop("Codice intervento")
data.pop("Accesso chirurgico")
data.pop("Codice diagnosi")
data.pop("Diagnosi")
data.pop("Note_comorbidita")
data.pop("Note_medicinali")
data.pop("Cormack")
data.pop("Mallampati")
data.pop("ASA")
data.pop("Diabete Mellito")
data.pop("NO_DIABETE_MELLITO")

# data["Tempo Tot BO Ormaweb"].fillna((data["Tempo Tot. SO OrmaWeb "] + data["Tempo Tot. RR "]), inplace=True)
# data["Tempo Tot BO Ormaweb"] = data["Tempo Tot BO Ormaweb"].replace("", (data["Tempo Tot. SO OrmaWeb "] + data["Tempo Tot. RR "]))
# data["Tempo Tot BO Ormaweb"].fillna((data["Tempo Tot. SO OrmaWeb "] + data["Tempo Tot. RR "]), inplace=True)
# data["Tempo Tot BO Ormaweb"].dropna(inplace=True)

for idx, row in data.iterrows():
    if np.isnan(row["Tempo Tot BO Ormaweb"]):
        if np.isnan(row["Tempo Tot. SO OrmaWeb"]) and np.isnan(row["Tempo Tot. RR"]):
            if not DATASET_GESTIONALE:
                data.drop(index=idx, inplace=True)
        else:
            data.loc[idx, "Tempo Tot BO Ormaweb"] = row["Tempo Tot. SO OrmaWeb"] + row["Tempo Tot. RR"]

data["BMI"] = data["Peso"] / (data["Altezza"] ** 2)

#print(data["feasible"].value_counts())

# data.to_csv("MLMED_Dataset.csv", index=False)
###################################################################################################################

### Split into validation and train ###

# ouped = data.groupby("54.21")

# for key, grp in gouped:
#    print(key, gouped.get_group(key)["54.21"])
if not DATASET_GESTIONALE:
    data = data.dropna()
data.pop('#N/D')
#print(data.shape)
data.pop("0.0")

data = data[data["INTERVENTI SULL’APPARATO DIGERENTE"] == 1]

to_drop = ['INTERVENTI SUL SISTEMA ENDOCRINO', 'INTERVENTI SULL’APPARATO URINARIO',
           'INTERVENTI SULL’APPARATO MUSCOLOSCHELETRICO',
           'INTERVENTI SUL SISTEMA EMATICO E LINFATICO',
           'INTERVENTI SUI TEGUMENTI',
           'INTERVENTI SUGLI ORGANI GENITALI FEMMINILI',
           'INTERVENTI SUL SISTEMA RESPIRATORIO',
           'INTERVENTI SUL SISTEMA CARDIOVASCOLARE',
           'INTERVENTI SUGLI ORGANI GENITALI MASCHILI',
           'MISCELLANEA DI PROCEDURE DIAGNOSTICHE E TERAPEUTICHE']

for elmnt in to_drop:
    if elmnt in data:
        data.pop(elmnt)

#print(data.columns)

if not DATASET_GESTIONALE:
    for i in tqdm(range(data.shape[0])):  # iterate over rows
        for j in range(data.shape[1]):  # iterate over columns
            value = data.iloc[i, j]  # get cell value
            try:
                if np.isnan(float(value)):
                    # print(value)
                     print(i, j)
            except:
                pass

    index_for_validation = []
    for key in count:
        occurence = count[key]
        if key in data:
            if occurence >= 6:
                to_take = math.ceil(0.05 * occurence)

                list_index = data.loc[data[key] == 1].index.values.tolist()
                random_extracted = random.choices(list_index, k=to_take)
                index_for_validation = index_for_validation + random_extracted
            else:
                data = data[data[key] != 1]
                data.pop(key)

    index_for_validation = list(set(index_for_validation))

    index_for_test = []
    for key in count:
        occurence = count[key]
        if occurence >= 4:
            to_take = math.ceil(0.1 * occurence)
            if key in data:
                list_index = data.loc[data[key] == 1].index.values.tolist()
                #print(key, list_index)
                random_extracted = random.choices(list_index, k=to_take)
                while not common_data(index_for_validation, random_extracted):
                    random_extracted = random.choices(list_index, k=to_take)
                index_for_test = index_for_test + random_extracted

    index_for_test = list(set(index_for_test))

    data.rename(columns=codes, inplace=True)
    #print("Numero di colonne:", len(list(data.columns)))

###########################################################################################################
# Prima di salvare il tutto dobbiamo riempire automaticamente i campi vuoti del Bluetooth che sono mancanti
# Prima riempiamo tutti i campi che sono a 0
# Poi controlliamo che: se la recovery in BL è 0 e in ormaweb è <10 allora non facciamo nulla, se il tempo BO del bluetooth è minore della somma di SO + RR dopo la sostituzione allora
# sostituiamo anche lui con i dati di ormaweb

for index, row in data.loc[data["feasible"] == False].iterrows():  # data.loc[data["feasible"] == False].iterrows():
    data.loc[index, "BLE_tot_BO_time"] = row["Tempo Tot BO Ormaweb"]
    data.loc[index, "BLE_tot_RR_time"] = row["Tempo Tot. RR"]
    data.loc[index, "BLE_tot_OR_time"] = row["Tempo Tot. SO OrmaWeb"]

for index, row in data.loc[data["feasible"] == True].iterrows():
    if row["BLE_tot_OR_time"] == 0:
        data.loc[index, "BLE_tot_OR_time"] = row["Tempo Tot. SO OrmaWeb"]
    if row["BLE_tot_RR_time"] == 0:
        if row["Tempo Tot. RR"] > 10:
            data.loc[index, "BLE_tot_RR_time"] = row["Tempo Tot. RR"]
    if row["BLE_tot_BO_time"] == 0:
        data.loc[index, "BLE_tot_BO_time"] = row["Tempo Tot BO Ormaweb"]
    else:
        if (row["BLE_tot_OR_time"] + row["BLE_tot_RR_time"]) > row["BLE_tot_BO_time"]:
            data.loc[index, "BLE_tot_BO_time"] = row["Tempo Tot BO Ormaweb"]

##aggiungiamo le sale e tutte le colonne che vuole laura

if DATASET_GESTIONALE:
    for idx, rw in data.iterrows():
        cod_an = rw["Codice alfa numerico"]

        data.loc[idx, "Sala"] = df.loc[df["Codice alfa numerico"] == cod_an]["Sala"].iloc[0]
        data.loc[idx, "Ingresso BO OrmaWeb"] = df.loc[df["Codice alfa numerico"] == cod_an]["Ingresso BO OrmaWeb"].iloc[
            0]
        data.loc[idx, "Sala pronta OrmaWeb"] = df.loc[df["Codice alfa numerico"] == cod_an]["Sala pronta OrmaWeb"].iloc[
            0]
        data.loc[idx, "Ingresso SO OrmaWeb"] = df.loc[df["Codice alfa numerico"] == cod_an]["Ingresso SO OrmaWeb"].iloc[
            0]
        data.loc[idx, "Inizio Anestesia OrmaWeb"] = \
            df.loc[df["Codice alfa numerico"] == cod_an]["Inizio Anestesia OrmaWeb"].iloc[0]
        data.loc[idx, "Inzio Chirurgia OrmaWeb"] = \
            df.loc[df["Codice alfa numerico"] == cod_an]["Inzio Chirurgia OrmaWeb"].iloc[0]
        data.loc[idx, "Fine Chir. OrmaWeb"] = df.loc[df["Codice alfa numerico"] == cod_an]["Fine Chir. OrmaWeb"].iloc[0]
        data.loc[idx, "Fine Anest. OrmaWeb"] = df.loc[df["Codice alfa numerico"] == cod_an]["Fine Anest. OrmaWeb"].iloc[
            0]
        data.loc[idx, "Uscita SO OrmaWeb"] = df.loc[df["Codice alfa numerico"] == cod_an]["Uscita SO OrmaWeb"].iloc[0]
        data.loc[idx, "Uscita BO OrmaWeb"] = df.loc[df["Codice alfa numerico"] == cod_an]["Uscita BO OrmaWeb"].iloc[0]
        data.loc[idx, "Ripristino SO OrmaWeb"] = \
            df.loc[df["Codice alfa numerico"] == cod_an]["Ripristino SO OrmaWeb"].iloc[0]

data.pop("INTERVENTI SULL’APPARATO DIGERENTE")
data.to_csv("MLMED_Dataset.csv", index=False)
data.to_excel("MLMED_Dataset.xlsx", index=False)

if not DATASET_GESTIONALE:
    validation = data.copy()
    test = data.copy()

    validation = validation[validation.index.isin(index_for_validation)]
    test = test[test.index.isin(index_for_test)]
    data.drop(index_for_validation + index_for_test, axis=0, inplace=True)

    data.to_csv("MLMED_Dataset_train.csv", index=False)
    validation.to_csv("ML_MED_Dataset_validation.csv", index=False)
    test.to_csv("ML_MED_Dataset_test.csv", index=False)
