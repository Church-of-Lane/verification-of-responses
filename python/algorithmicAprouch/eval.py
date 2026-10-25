import symbolic

def printMetrics(title, results):
    count_correct,count_wrong,count_wrong_spam,count_wrong_ham,actual_ham,actual_spam = results

    # positive class = spam
    FN = count_wrong_spam          # spam predicted as ham
    FP = count_wrong_ham           # ham predicted as spam
    TP = actual_spam - FN
    TN = actual_ham - FP
    total = actual_ham + actual_spam

    def div(a, b): return a / b if b else 0.0

    accuracy = div(TP + TN, total)
    precision = div(TP, TP + FP)
    recall = div(TP, TP + FN)              # sensitivity / true positive rate
    specificity = div(TN, TN + FP)         # true negative rate
    f1 = div(2 * precision * recall, precision + recall)
    fpr = div(FP, FP + TN)
    fnr = div(FN, FN + TP)
    npv = div(TN, TN + FN)                 # negative predictive value
    balanced_acc = (recall + specificity) / 2

    print(f"=== {title} (positive = spam) ===")
    print(f"Total samples : {total}  (spam: {actual_spam}, ham: {actual_ham})")
    print(f"Correct       : {count_correct}")
    print(f"Wrong         : {count_wrong}")
    print()
    print("Confusion matrix:")
    print(f"{'':>15}{'pred spam':>12}{'pred ham':>12}")
    print(f"{'actual spam':>15}{TP:>12}{FN:>12}")
    print(f"{'actual ham':>15}{FP:>12}{TN:>12}")
    print()
    print(f"TP: {TP}  FP: {FP}  TN: {TN}  FN: {FN}")
    print()
    print(f"Accuracy          : {accuracy:.4f}")
    print(f"Precision         : {precision:.4f}")
    print(f"Recall            : {recall:.4f}")
    print(f"F1 score          : {f1:.4f}")
    print(f"Specificity       : {specificity:.4f}")
    print(f"Balanced accuracy : {balanced_acc:.4f}")
    print(f"False pos. rate   : {fpr:.4f}")
    print(f"False neg. rate   : {fnr:.4f}")
    print(f"Neg. pred. value  : {npv:.4f}")
    print()

def evalSymbolicData():
    printMetrics("Symbolic classifier evaluation", symbolic.main())

def evalHumanData():
    printMetrics("Human baseline - Kirill", symbolic.forHuman("Kirill"))
    printMetrics("Human baseline - Roman", symbolic.forHuman("Roman"))
    printMetrics("Symbolic classifier on human baseline set", symbolic.forHumanSet())


if __name__ == "__main__":
    evalSymbolicData()
    evalHumanData()
