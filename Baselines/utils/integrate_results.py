import pandas as pd
import numpy as np
from tqdm import tqdm
from evaluation import score_aggregates


def integrate_results():
    f0 = '/home/marzieh/Desktop/OptimaServer/exchange/Marzieh/Mario/log_mario_test/swin_base_patch4_window7_224_224_Mario/20240724-151722/fold_0/test_results.csv'
    f1 = '/home/marzieh/Desktop/OptimaServer/exchange/Marzieh/Mario/log_mario_test/swin_base_patch4_window7_224_224_Mario/20240724-153414/fold_1/test_results.csv'
    f2 = '/home/marzieh/Desktop/OptimaServer/exchange/Marzieh/Mario/log_mario_test/swin_base_patch4_window7_224_224_Mario/20240724-160108/fold_2/test_results.csv'
    f3 = '/home/marzieh/Desktop/OptimaServer/exchange/Marzieh/Mario/log_mario_test/swin_base_patch4_window7_224_224_Mario/20240724-164030/fold_3/test_results.csv'
    f4 = '/home/marzieh/Desktop/OptimaServer/exchange/Marzieh/Mario/log_mario_test/swin_base_patch4_window7_224_224_Mario/20240725-101441/fold_4/test_results.csv'
    f_1 = '/home/marzieh/bigData/Mario/Logfirst_try/log_mario_test/swin_base_patch4_window7_224_224_Mario/20240618-142504/test_results.csv'

    f0 = pd.read_csv(f0)
    f1 = pd.read_csv(f1)
    f2 = pd.read_csv(f2)
    f3 = pd.read_csv(f3)
    f4 = pd.read_csv(f4)
    f_1 = pd.read_csv(f_1)

    final = []  # list of tuples (case, prediction)
    for case in list(f0['case']):
        a = [f0.loc[f0["case"] == case]['prediction'].values[0],
             f1.loc[f1["case"] == case]['prediction'].values[0],
             f2.loc[f2["case"] == case]['prediction'].values[0],
             f3.loc[f3["case"] == case]['prediction'].values[0],
             f4.loc[f4["case"] == case]['prediction'].values[0],
             f_1.loc[f_1["case"] == case]['prediction'].values[0]
             ]
        res = max(set(a), key=a.count)
        # if res = 1 then take the second max value
        if res == 1:
            a.remove(res)
            res = max(set(a), key=a.count)
        final.append((case, res))
    df = pd.DataFrame(final, columns=['case', 'prediction'])
    df.to_csv('results.csv', index=False)


def prediction_evaluation(path):
    data = pd.read_csv(path + '/test_results.csv')

    "case,prediction"
    predictions = data['prediction']
    unique, counts = np.unique(predictions, return_counts=True)
    # draw a pie chart based on the predictions
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.pie(counts, labels=unique, autopct='%1.1f%%')
    plt.savefig(path + '/pie_chart.png')
    plt.clf()
    plt.close()


def check_cross_validation():
    pathes = {0: '/home/marzieh/Desktop/OptimaServer/exchange/Marzieh/Mario/annotations/task1_fold_0.csv',
              1: '/home/marzieh/Desktop/OptimaServer/exchange/Marzieh/Mario/annotations/task1_fold_1.csv',
              2: '/home/marzieh/Desktop/OptimaServer/exchange/Marzieh/Mario/annotations/task1_fold_2.csv'}
    results_pathes = {
        0: '/home/marzieh/Desktop/OptimaServer/exchange/Marzieh/Mario/log_mario_test/swin_base_patch4_window7_224_224_Mario/20240621-163101/fold_0/',
        1: '/home/marzieh/Desktop/OptimaServer/exchange/Marzieh/Mario/log_mario_test/swin_base_patch4_window7_224_224_Mario/20240621-162128/fold_1/',
        2: '/home/marzieh/Desktop/OptimaServer/exchange/Marzieh/Mario/log_mario_test/swin_base_patch4_window7_224_224_Mario/20240621-155121/fold_2/'}

    for i in tqdm(range(3)):
        predictions = []
        data = pd.read_csv(pathes[i])
        results = pd.read_csv(results_pathes[i] + '/test_results.csv')
        for id, sample in data.iterrows():
            pred = results[results['case'] == sample['case']]['prediction']
            sample['prediction'] = pred
            predictions.append(sample)
        predictions = pd.DataFrame(data=predictions)
        prediction_evaluation(results_pathes[i])
        score_aggregates(list(predictions['label']), list(predictions['prediction']))
        # draw confusion matrix
        from sklearn.metrics import confusion_matrix
        import seaborn as sns
        import matplotlib.pyplot as plt
        cm = confusion_matrix(list(predictions['label']), list(predictions['prediction']))
        sns.heatmap(cm, annot=True, fmt='g')
        plt.savefig(results_pathes[i] + '/confusion_matrix.png')
        plt.clf()
        plt.close()


if __name__ == '__main__':
    integrate_results()
