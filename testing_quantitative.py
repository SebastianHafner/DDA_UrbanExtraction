# TODO: write this to file
def inference_regional_testing_sn7(config_name: str, threshold: float):
    print(f'{"-" * 10} {config_name} {"-" * 10}')

    data = get_quantitative_data_sn7(config_name)
    for metric in ['f1_score', 'precision', 'recall']:
        print(metric)
        region_values = []
        for region_name, region in data.items():

            y_true = np.concatenate([site['y_true'] for site in region], axis=0)
            y_prob = np.concatenate([site['y_prob'] for site in region], axis=0)

            if metric == 'f1_score':
                value = metrics.f1_score_from_prob(y_prob, y_true, threshold)
            elif metric == 'precision':
                value = metrics.precsision_from_prob(y_prob, y_true, threshold)
            else:
                value = metrics.recall_from_prob(y_prob, y_true, threshold)

            print(f'{region_name}: {value:.3f},', end=' ')
            region_values.append(value)

        print('')
        min_ = np.min(region_values)
        max_ = np.max(region_values)
        mean = np.mean(region_values)
        std = np.std(region_values)

        print(f'summary statistics: {min_:.3f} min, {max_:.3f} max, {mean:.3f} mean, {std:.3f} std')

    y_true = np.concatenate([site['y_true'] for region in data.values() for site in region], axis=0)
    y_prob = np.concatenate([site['y_prob'] for region in data.values() for site in region], axis=0)
    f1 = metrics.f1_score_from_prob(y_prob, y_true, 0.5)
    prec = metrics.precsision_from_prob(y_prob, y_true, 0.5)
    rec = metrics.recall_from_prob(y_prob, y_true, 0.5)
    print(f'total: {f1:.3f} f1 score, {prec:.3f} precision, {rec:.3f} recall')

if __name__ == '__main__':
    pass