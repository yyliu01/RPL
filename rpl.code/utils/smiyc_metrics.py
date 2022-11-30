import matplotlib.pyplot as plt
import torch
import numpy
from easydict import EasyDict


class SmiycMeasures:
    def __init__(self):
        self.num_bins = 768
        self.bin_strategy = 'percentiles'

    def __call__(self, anomaly_p, label_pixel_gt):
        mask_roi = label_pixel_gt < 255
        predictions_in_roi = anomaly_p[mask_roi]
        labels_in_roi = label_pixel_gt[mask_roi]
        bc = self.binary_confusion_matrix(
            prob=predictions_in_roi,
            gt_label_bool=labels_in_roi.astype(bool),
            num_bins=self.num_bins,
            bin_strategy=self.bin_strategy,
        )
        return bc

    def aggregate(self, results):
        ag = self.aggregate_dynamic_bins(results)
        thresholds = ag.thresholds
        cmats = ag.cmat
        curves = self.curves_from_cmats(cmats, thresholds)
        return curves

    @ staticmethod
    def binary_confusion_matrix(
            prob: numpy.ndarray, gt_label_bool: numpy.ndarray,
            num_bins: int = 1024, bin_strategy='uniform',  # : Literal['uniform', 'percentiles'] = 'uniform',
            normalize: bool = False, dtype=numpy.float64):

        area = gt_label_bool.__len__()

        gt_area_true = numpy.count_nonzero(gt_label_bool)
        gt_area_false = area - gt_area_true

        prob_at_true = prob[gt_label_bool]
        prob_at_false = prob[~gt_label_bool]

        if bin_strategy == 'uniform':
            # bins spread uniforms in 0 .. 1
            bins = num_bins
            histogram_range = [0, 1]

        elif bin_strategy == 'percentiles':
            # dynamic bins representing the range of occurring values
            # bin edges are following the distribution of positive and negative pixels

            bins = [
                [0, 1],  # make sure 0 and 1 are included
            ]

            if prob_at_true.size:
                bins += [
                    numpy.quantile(prob_at_true, numpy.linspace(0, 1, min(num_bins // 2, prob_at_true.size))),
                ]
            if prob_at_false.size:
                bins += [
                    numpy.quantile(prob_at_false, numpy.linspace(0, 1, min(num_bins // 2, prob_at_false.size))),
                ]

            bins = numpy.concatenate(bins)

            # sort and remove duplicates, duplicated cause an exception in numpy.histogram
            bins = numpy.unique(bins)

            histogram_range = None

        # the area of positive pixels is divided into
        #	- true positives - above threshold
        #	- false negatives - below threshold
        tp_rel, _ = numpy.histogram(prob_at_true, bins=bins, range=histogram_range)
        # the curve goes from higher thresholds to lower thresholds
        tp_rel = tp_rel[::-1]
        # cumsum to get number of tp at given threshold
        tp = numpy.cumsum(tp_rel)
        # GT-positives which are not TP are instead FN
        fn = gt_area_true - tp

        # the area of negative pixels is divided into
        #	- false positives - above threshold
        #	- true negatives - below threshold
        fp_rel, bin_edges = numpy.histogram(prob_at_false, bins=bins, range=histogram_range)
        # the curve goes from higher thresholds to lower thresholds
        bin_edges = bin_edges[::-1]
        fp_rel = fp_rel[::-1]
        # cumsum to get number of fp at given threshold
        fp = numpy.cumsum(fp_rel)
        # GT-negatives which are not FP are instead TN
        tn = gt_area_false - fp

        cmat_sum = numpy.array([
            [tp, fp],
            [fn, tn],
        ]).transpose(2, 0, 1).astype(dtype)

        # cmat_rel = numpy.array([
        # 	[tp_rel, fp_rel],
        # 	[-tp_rel, -fp_rel],
        # ]).transpose(2, 0, 1).astype(dtype)

        if normalize:
            cmat_sum *= (1. / area)
        # cmat_rel *= (1./area)

        return EasyDict(
            bin_edges=bin_edges,
            cmat_sum=cmat_sum,
            # cmat_rel = cmat_rel,
            tp_rel=tp_rel,
            fp_rel=fp_rel,
            num_pos=gt_area_true,
            num_neg=gt_area_false,
        )


    @ staticmethod
    def get_no_prediction_prefix(cmats):
        """
        The threshold goes from high to low
        At the beginning, we have 0 predictions and there is no valid precision
        Remove the prefix with 0 predictions

        This function returns the number of elements to remove from the beginning.
        """

        for i in range(cmats.__len__()):
            if cmats[i, 0, 0] + cmats[i, 0, 1] > 0.01:
                return i

        raise ValueError('No predictions made at all')

    def curves_from_cmats(self, cmats, thresholds):
        # The threshold goes from high to low
        # At the beginning, we have 0 predictions and there is no valid precision
        # Remove the prefix with 0 predictions

        num_remove = self.get_no_prediction_prefix(cmats)

        if num_remove > 0:
            print(f'Skip {num_remove}')
            cmats = cmats[num_remove:]
            thresholds = thresholds[num_remove:]

        tp = cmats[:, 0, 0]
        fp = cmats[:, 0, 1]
        fn = cmats[:, 1, 0]
        tn = cmats[:, 1, 1]

        tp_rates = tp / (tp + fn)
        fp_rates = fp / (fp + tn)

        precisions = tp / (tp + fp)
        recalls = tp / (tp + fn)
        f1_scores = (2 * tp) / (2 * tp + fp + fn)

        tpr95_index = numpy.searchsorted(tp_rates, 0.95)
        if tpr95_index < tp_rates.shape[0]:
            fpr_tpr95 = fp_rates[tpr95_index]
            tpr95_threshold = float(thresholds[tpr95_index])
        else:
            # tpr95 was not reached
            fpr_tpr95 = 1.0
            tpr95_threshold = 0.0

        recall50_index = numpy.searchsorted(recalls, 0.50)
        recall50_threshold = float(thresholds[recall50_index])

        ix = numpy.nanargmax(f1_scores)
        best_f1_threshold = float(thresholds[ix])
        best_f1 = f1_scores[ix]

        print(
            'ap-sum', numpy.sum(numpy.diff(recalls) * precisions[:-1]),
            'ap-trapz', numpy.trapz(precisions, recalls),
        )

        return EasyDict(
            # curves
            curve_tpr=tp_rates,
            curve_fpr=fp_rates,
            curve_precision=precisions,
            curve_recall=recalls,

            thresholds=thresholds,

            # areas
            area_ROC=numpy.trapz(tp_rates, fp_rates),
            area_PRC=numpy.trapz(precisions, recalls),

            tpr95_fpr=fpr_tpr95,
            tpr95_threshold=tpr95_threshold,

            recall50_threshold=recall50_threshold,
            best_f1_threshold=best_f1_threshold,
            best_f1=best_f1

        )

    @staticmethod
    def aggregate_dynamic_bins(frame_results):
        thresholds = numpy.concatenate([r.bin_edges[1:] for r in frame_results])

        tp_relative = numpy.concatenate([r.tp_rel for r in frame_results], axis=0)
        fp_relative = numpy.concatenate([r.fp_rel for r in frame_results], axis=0)

        num_positives = sum(r.num_pos for r in frame_results)
        num_negatives = sum(r.num_neg for r in frame_results)

        threshold_order = numpy.argsort(thresholds)[::-1]

        # We start at threshold = 1, and lower it
        # Initially, prediction=0, all GT=1 pixels are false-negatives, and all GT=0 pixels are true-negatives.

        tp_cumu = numpy.cumsum(tp_relative[threshold_order].astype(numpy.float64))
        fp_cumu = numpy.cumsum(fp_relative[threshold_order].astype(numpy.float64))

        cmats = numpy.array([
            # tp, fp
            [tp_cumu, fp_cumu],
            # fn, tn
            [num_positives - tp_cumu, num_negatives - fp_cumu],
        ]).transpose([2, 0, 1])

        return EasyDict(
            cmat=cmats,
            thresholds=thresholds[threshold_order],
        )

