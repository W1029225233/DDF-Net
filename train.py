import time
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from util.metrics import PSNR, SSIM

import torch
import numpy as np
import random


# 设置随机种子
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train(opt, data_loader, model, visualizer):
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    print('#training images = %d' % dataset_size)
    total_steps = 0

    # 如果继续训练，重新设置学习率
    if opt.continue_train:
        new_lr = 1e-4
        for param_group in model.optimizer_G.param_groups:
            param_group['lr'] = new_lr
        for param_group in model.optimizer_D.param_groups:
            param_group['lr'] = new_lr
        print(f"Learning rate reset to {new_lr}")

    if opt.continue_train:
        start_epoch = int(opt.which_epoch) + 1
    else:
        start_epoch = opt.epoch_count

    total_epochs = opt.niter + opt.niter_decay

    for epoch in range(start_epoch, total_epochs + 1):
        epoch_start_time = time.time()
        epoch_iter = 0

        # 高斯去噪不需要resample，因为是在线随机加噪
        # 同一张图每次遇到的噪声都不同，所以用全部数据即可
        # 删除或注释掉原来的resample代码：
        # if hasattr(data_loader.dataset, 'resample'):
        #     data_loader.dataset.resample()
        #     print(f'[Epoch {epoch}] Resampled: using {len(data_loader.dataset)} images')

        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            total_steps += opt.batchSize
            epoch_iter += opt.batchSize

            model.set_input(data)
            model.optimize_parameters()

            if total_steps % opt.display_freq == 0:
                results = model.get_current_visuals()
                psnrMetric = PSNR(results['fake_B'], results['real_B'])
                print('PSNR on Train = %f' % psnrMetric)
                visualizer.display_current_results(results, epoch)

            if total_steps % opt.print_freq == 0:
                errors = model.get_current_errors()
                t = (time.time() - iter_start_time) / opt.batchSize
                visualizer.print_current_errors(epoch, epoch_iter, errors, t)
                if opt.display_id > 0:
                    # 注意：这里的dataset_size可能变了，用实际长度
                    visualizer.plot_current_errors(epoch, float(epoch_iter) / dataset_size, opt, errors)

            if total_steps % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
                model.save('latest')

        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))
            model.save('latest')
            model.save(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, total_epochs, time.time() - epoch_start_time))

        if epoch > opt.niter:
            model.update_learning_rate()


# 设置随机种子
set_random_seed(666)
opt = TrainOptions().parse()
data_loader = CreateDataLoader(opt)
model = create_model(opt)
visualizer = Visualizer(opt)

if opt.continue_train:
    opt.epoch_count = int(opt.which_epoch) + 1

train(opt, data_loader, model, visualizer)