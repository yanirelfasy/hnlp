import pandas as pd
from pathlib import Path
from sklearn.metrics import confusion_matrix,precision_score
from sklearn.metrics import recall_score,f1_score,accuracy_score

class Evaluator:
    def __init__(self, test_df):
        self.test_df=test_df
        print(f"test_df has {len(self.test_df)} rows")
#                            header=None, 
#                            names=["comment","label"]).reset_index()

    def get_true_labels(self):
        return self.test_df[["index","label"]]

    def show_errors(self, predicted_labels, n, correct_label):
        pred_df=self.test_df.copy()
        pred_df.loc[:,"pred_label"]=predicted_labels
        wrong_df=pred_df[pred_df.pred_label!=pred_df.label]
        wrong_df=wrong_df[wrong_df.label==correct_label]
        print(f"overall, there are {len(wrong_df)} instances with wrong kind")
        num_shown=min(n, len(wrong_df))
        for i in range(0, num_shown):
            cur_row=wrong_df.iloc[i]
            print(cur_row["text"], "true: ", cur_row["label"], "predicted: ", cur_row["pred_label"] + "\n")
            # print(f"{cur_row.Censored_Desc}, true:{cur_row.label}, predicted:{cur_row.pred_label}\n")

    def show_correct(self, predicted_labels, n):
        pred_df=self.test_df.copy()
        pred_df.loc[:,"pred_label"]=predicted_labels
        right_df=pred_df[pred_df.pred_label==pred_df.label]
        print(f"overall, there are {len(right_df)} instances with right kind")
        num_shown=min(n, len(right_df))
        for i in range(0, num_shown):
            cur_row=right_df.iloc[i]
            print(f"{cur_row.text}, true:{cur_row.label}, predicted:{cur_row.pred_label}\n")

    def evaluate(self, predicted_labels):
        prec_mic=precision_score(self.test_df.label, 
                                predicted_labels,
                                average="micro")

        rec_mic=recall_score(self.test_df.label, 
                                predicted_labels,
                                average="micro")

        f1_mic=f1_score(self.test_df.label, 
                        predicted_labels,
                        average="micro")
        print(f"Micro precision:{prec_mic}, recall:{rec_mic}, f1:{f1_mic}")
        prec_mac=precision_score(self.test_df.label, 
                                predicted_labels,
                                average="macro")

        rec_mac=recall_score(self.test_df.label, 
                                predicted_labels,
                                average="macro")

        f1_mac=f1_score(self.test_df.label, 
                        predicted_labels,
                        average="macro")
        print(f"Macro precision:{prec_mac}, recall:{rec_mac}, f1:{f1_mac}")


        acc=accuracy_score(self.test_df.label, 
                            predicted_labels)
        labels = ["netanyahu", "meravmichaelijs", "naftalibennett", "tamarzandberg", "yairlapid", "ayelet__shaked", "AvigdorLiberman", "regev_miri", "NitzanHorowitz"]
        print(f"Accuracy: {acc}")
        cm=confusion_matrix(self.test_df.label,
                            predicted_labels,
                            labels = labels
                            )
        print(cm)
        f1_all = f1_score(self.test_df.label, 
                        predicted_labels,
                        average=None,
                        labels=labels)
        prec_all=precision_score(self.test_df.label, 
                        predicted_labels,
                        average=None,
                        labels=labels)

        rec_all=recall_score(self.test_df.label, 
                                predicted_labels,
                                average=None,
                                labels=labels)

        for index, label in enumerate(labels):
            print("LABEL: ", label, "F1: ", f1_all[index], " PREC: ", prec_all[index], " REC: ", rec_all[index])
        

def main():
    evaluator = Evaluator()
    print("true_labels distribution:")
    print(evaluator.get_true_labels()["label"].value_counts())
    print("hey")
    #read base file
    #call classifier and get results
    #calculate confusion matrix, precision, recall, F1, AUC
    #print results

if __name__=="__main__":
    main()

