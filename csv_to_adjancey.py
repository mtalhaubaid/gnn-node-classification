
# print(df.head(50))

# pre=df['Prereqs'].to_numpy().tolist()
# print(pre)
# # pre.replace('''''','')
# # print(pre)
# # src=ed['Prereqs'].to_numpy().tolist()
# print(len(df['SkillId']))
#
# for i in len(df):
#     df['SkillId'][1]
import pandas as pd
preReq=pd.read_excel(r"C:\Users\tubaid\PycharmProjects\smiple_GNN\GraphNeuralNet-master\skils.xlsx")
skID = list(preReq['SkillId'])
a_string = preReq['Prereqs'].fillna('0')

pReq = []
tmp = []
for i in range(len(a_string)):
    a_string[i] = a_string[i].replace(",", " ")
#     print(f'String {i} = {a_string[i]}')
    if len(a_string[i])>6:
        for word in a_string[i].split():
            if word.isdigit():
                tmp.append(int(word))
        pReq.append(tmp)
        tmp=[]
    else:
        for word in a_string[i].split():
            if word.isdigit():
                pReq.append([int(word)])

gS_Data = []

# print(gS_Data)
gPR_Data = []
for i in range(len(skID)):
    for j in range (len(pReq[i])):
        gS_Data.append(skID[i])
        gPR_Data.append(pReq[i][j])

gDf = pd.DataFrame({
    'SkillId':gS_Data,
    'PreRequisits':gPR_Data
})
print(gDf)
print(len(gPR_Data))
print(len(gS_Data))
