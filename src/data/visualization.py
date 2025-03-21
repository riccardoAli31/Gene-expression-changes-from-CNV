from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .dataset import CnvDataset
from plotnine import (
    ggplot,
    ggsave,
    aes,
    labs,
    ylim,
    geom_col,
    geom_point,
    facet_wrap
)
import matplotlib.pyplot as plt


def dataset_dist(dataset: CnvDataset):
    return torch.vstack([torch.hstack([e.sum(axis=1), t]) for e, t in dataset])


def dataset_tensor(dataset: CnvDataset, rows=None):
    rows = list(range(dataset[0][0].size()[0])) if rows == None else rows
    assert all(map(lambda x: x < dataset[0][0].size()[0], rows))
    data_tensor = torch.zeros((len(rows), dataset[0][0].size()[1]))
    for e, _ in dataset:
        data_tensor += e[rows,:]
    return data_tensor


def dataset_metadata(dist_tensor: torch.Tensor, emb_length=10_000):
    # list elements: 1. no, 2. some, 3. all positions in embedding show signal
    status_matrix = np.zeros(shape=(6, 3))
    # high_atac = [0] * 3
    # low_atac = [0] * 3
    # high_CNV-_loss, low_CNV-_loss = [0] * 3, [0] * 3
    # high_cnv_gain, low_cnv_gain = [0] * 3, [0] * 3
    # no_atac, some_atac, full_atac = 0, 0, 0
    # no_cnv_loss, some_cnv_loss, full_cnv_loss = 0, 0, 0
    # no_cnv_gain, some_cnv_gain, full_cnv_gain = 0, 0, 0
    for atac, cnv_loss, cnv_gain, target_class in dist_tensor[:,4:]:
        target_class = int(target_class)
        if atac == 0:
            status_matrix[target_class, 0] += 1
        elif atac == emb_length:
            status_matrix[target_class, 2] += 1
        else:
            status_matrix[target_class, 1] += 1

        if cnv_loss == 0:
            status_matrix[(2 + target_class), 0] += 1
        elif cnv_loss == emb_length:
            status_matrix[(2 + target_class), 2] += 1
        else:
            status_matrix[(2 + target_class), 1] += 1

        if cnv_gain == 0:
            status_matrix[(4 + target_class), 0] += 1
        elif cnv_gain == emb_length:
            status_matrix[(4 + target_class), 2] += 1
        else:
            status_matrix[(4 + target_class), 1] += 1
    
    # return dataframe for plotting
    return pd.DataFrame({
        'data': [
            'ATAC', 'ATAC', 'ATAC', 'ATAC', 'ATAC', 'ATAC',
            'CNV-', 'CNV-', 'CNV-', 'CNV-', 'CNV-', 'CNV-',
            'CNV+', 'CNV+', 'CNV+', 'CNV+', 'CNV+', 'CNV+'
            ],
        'classification': ['high', 'high', 'high', 'low', 'low', 'low'] * 3,
        'status': ['none', 'part', 'full'] * 6,
        'value': status_matrix.flatten()
        })


def plot_signal_strength(dataset: CnvDataset, batch: int, split: str, out_path: Path):
    assert batch in (1, 2), 'No valid batch'
    # assert split in ('train', 'val', 'test'), 'No valid split name'

    dataset_dist_tensor = dataset_dist(dataset)
    dataset_meta_df = dataset_metadata(dataset_dist_tensor)

    class_balance_df = pd.DataFrame(dataset.class_balance())
    class_balance_df['status'] = class_balance_df.index
    class_balance_df['classification'] = 'class balance'
    class_balance_df['data'] = ''
    class_balance_df

    p = ggplot(data=dataset_meta_df, mapping=aes(x='data', y='value')) +\
        geom_col(aes(fill='status'), position='stack') +\
        geom_col(data=class_balance_df, mapping=aes(y='count', fill='status'), position='dodge') +\
        facet_wrap('~classification', scales='free_x') +\
        labs(
            title='Batch {} {} signal summary'.format(batch, split),
            x='Signal type', y='# data points')
    ggsave(p, out_path / 'batch{}_{}_signal_summary.png'.format(batch, split))
    p.show()

    return dataset_dist_tensor, dataset_meta_df


def plot_signal_localization(dataset: CnvDataset, out_path: Path, name: str):
    assert out_path.is_dir(), 'Improper output directory: {}'.format(out_path)

    signal_tensor = dataset_tensor(dataset)

    # DNA base signal
    plot_df = pd.DataFrame({
        'pos': list(range(10000)) * 4,
        'base': ['A'] * 10000 + ['C'] * 10000 + ['G'] * 10000 + ['T'] * 10000,
        'count': np.hstack([signal_tensor[i,:] for i in range(4)]),
        'percent': np.hstack([signal_tensor[i,:] for i in range(4)]) / len(dataset)
    })
    p = ggplot(plot_df, aes(x='pos', y='count', fill='base')) +\
            geom_col() +\
            labs(
                title='{} (size: {}) DNA signal localization'.format(name, str(len(dataset))),
                x='Position in Embedding', y='Count'
            )
    ggsave(p, out_path / '{}_DNA_signal_localization_count.png'.format(name))
    p.show()
    p = ggplot(plot_df, aes(x='pos', y='percent', fill='base')) +\
            geom_col() +\
            labs(
                title='{} (size: {}) DNA signal localization'.format(name, str(len(dataset))),
                x='Position in embedding',
                y='Signal strength'
            )
    ggsave(p, out_path / '{}_DNA_signal_localization_percent.png'.format(name))
    p.show()

    # open chromatin signal
    plot_df = pd.DataFrame({
        'pos': list(range(10000)),
        'count': signal_tensor[4,:],
        'percent': signal_tensor[4,:] / len(dataset)
    })
    p = ggplot(plot_df, aes(x='pos', y='count')) +\
            geom_col() +\
            labs(
                title='{} (size: {}) open chromatin signal localization'.format(name, str(len(dataset))),
                x='Position in Embedding', y='Count'
            )
    ggsave(p, out_path / '{}_ATAC_signal_localization_count.png'.format(name))
    p.show()
    p = ggplot(plot_df, aes(x='pos', y='percent')) +\
            geom_col() + ylim(0, 1) +\
            labs(
                title='{} (size: {}) open chromatin signal localization'.format(name, str(len(dataset))),
                x='Position in embedding',
                y='Signal strength'
            )
    ggsave(p, out_path / '{}_ATAC_signal_localization_percent.png'.format(name))
    p.show()
    
    # CNV loss / gain signal
    plot_df = pd.DataFrame({
        'pos': list(range(10000)) * 2,
        'CNV status': ['loss'] * 10000 + ['gain'] * 10000,
        # 'count': np.hstack([signal_tensor[i,:] for i in range(5,7)]),
        'percent': np.hstack([signal_tensor[i,:] for i in range(5,7)]) / len(dataset)
    })
    # p = ggplot(plot_df, aes(x='pos', y='count', fill='CNV status')) +\
    #         geom_col() +\
    #         labs(
    #             title='{} CNV signal localization'.format(name),
    #             x='Position in Embedding', y='Count'
    #         )
    # ggsave(p, out_path / '{}_CNV_signal_localization_count.png'.format(name))
    # p.show()
    p = ggplot(plot_df, aes(x='pos', y='percent', fill='CNV status')) +\
            geom_col() + ylim((0, 1)) +\
            labs(
                title='{} (size: {}) CNV signal localization'.format(name, str(len(dataset))),
                x='Position in embedding',
                y='Signal strength'
            )
    ggsave(p, out_path / '{}_CNV_signal_localization_percent.png'.format(name))
    p.show()


def plot_signal_heatmap(dataset: CnvDataset):
    signal_tensor = dataset_tensor(dataset=dataset)

    # plot heatmap
    plt.imshow(signal_tensor.numpy(), cmap='Blues', interpolation='nearest')


def plot_atac_expr_scatter(dataset: CnvDataset, out_path: Path, name: str):
    # TODO: assert regression dataset
    dataset_tensor = dataset_dist(dataset)

    plot_df = pd.DataFrame({
        'ATAC': dataset_tensor[:,4],
        'expression': dataset_tensor[:,7],
        'loss': dataset_tensor[:,5],
        'gain': dataset_tensor[:,6],
        'CNV status': 'normal'
    })
    plot_df.loc[plot_df['loss'] > 0, 'CNV status'] = 'loss'
    plot_df.loc[plot_df['gain'] > 0, 'CNV status'] = 'gain'
    plot_df

    p = ggplot(
        data=plot_df,
        mapping=aes(x='ATAC', y='expression', color='CNV status')
        ) + geom_point(alpha=0.3) + labs(
            title=name,
            x='# open chromatin positions',
            y='Gene expression'
        )
    ggsave(p, out_path / '{}_atac_expr_corr.png'.format(name))
    p.show()
