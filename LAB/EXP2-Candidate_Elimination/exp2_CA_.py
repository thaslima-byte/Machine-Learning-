print("J THASLIMA NASREEN \n(192424152)")
import pandas as pd
# Read CSV file
df = pd.read_csv("find s  - Sheet1 (1).csv")
# Rename the last column
df.rename(columns={df.columns[-1]: "Allergy"}, inplace=True)
print("Training Data\n")
print(df)
# Ignore Number column
concepts = df.iloc[:, 1:-1].values
target = df.iloc[:, -1].values
# Initialize S and G
S = ['Ø'] * len(concepts[0])
G = [['?'] * len(concepts[0])]
print("\nInitial Specific Boundary (S):", S)
print("Initial General Boundary (G):", G)
for i, h in enumerate(concepts):


    if target[i] == "Yes":


        if S == ['Ø'] * len(concepts[0]):
            S = list(h)
        else:
            for j in range(len(S)):
                if S[j] != h[j]:
                    S[j] = '?'


        G = [g for g in G if all(g[k] == '?' or g[k] == h[k] for k in range(len(g)))]


    else:


        new_G = []


        for g in G:
            for j in range(len(g)):
                if g[j] == '?':
                    if S[j] != '?' and S[j] != h[j]:
                        new_hypothesis = g.copy()
                        new_hypothesis[j] = S[j]


                        if new_hypothesis not in new_G:
                            new_G.append(new_hypothesis)


        G = new_G


    print("\nAfter Example", i + 1)
    print("S =", S)
    print("G =", G)


print("\nFinal Specific Boundary (S):", S)
print("Final General Boundary (G):", G)
