class Ensemble(object):
    def __init__(self, method, stage='training'):
        self.method = method

    def load_ensemble(self, expname):
        pass

    def predict(self, split, models):
        """
        models should be a lit of (model_folder,expname,weights)
        """
        preds = []
        for folder, expname, alpha in models:
            pred_path = ojoin(folder, '{}_{}_preds.h5'.format(expname, split))
            pd.read_hdf(pred_path)
            preds.append(preds)

        return preds
