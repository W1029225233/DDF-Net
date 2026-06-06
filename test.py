import time
import os
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from util import html
from util.metrics import PSNR
# from ssim import ssim as SSIM
from PIL import Image

opt = TestOptions().parse() # 创建TestOptions实例，并解析命令行参数或配置文件中的选项。
# 设置线程数nThreads为1，因为测试代码只支持单线程。
opt.nThreads = 1  # test code only supports nThreads = 1
# 设置批量大小batchSize为1，因为测试代码只支持单张图像的测试。
opt.batchSize = 1  # test code only supports batchSize = 1
# 设置serial_batches为True，表示不打乱数据顺
opt.serial_batches = True  # no shuffle
# 设置no_flip为True，表示在测试过程中不进行图像翻
opt.no_flip = True  # no flip

# 使用CreateDataLoader创建数据加载器，并加载测试数据集。
# 使用create_model函数创建模型实例。
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
model = create_model(opt)

# 创建Visualizer实例，用于保存和展示结果图像。
# 创建HTML页面，用于展示测试结果。页面的路径由选项中的results_dir、name和当前的phase、which_epoch组成。
visualizer = Visualizer(opt)
# create website
web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
# test
total_time = 0.0
image_times = []

# 遍历测试数据集中的每一张图像。
# 如果处理的图像数量达到opt.how_many指定的数量，则停止测试。
for i, data in enumerate(dataset):
    if i >= opt.how_many:
        break
    # 对于每张图像，记录开始处理的时间。
    start_time = time.time()  # Start time for each image
    # 将图像数据输入模型
    model.set_input(data)
    # 执行测试
    model.test()
    # 获取模型输出的可视化结果。
    visuals = model.get_current_visuals()
    # 获取当前处理的图像路径，并打印出来。
    img_path = model.get_image_paths()
    print('process image... %s' % img_path)
    # 使用visualizer保存结果图像到HTML页面。
    visualizer.save_images(webpage, visuals, img_path)
    # 记录处理每张图像结束的时间，并计算处理每张图像所需的时间。
    end_time = time.time()  # End time for each image
    elapsed_time = end_time - start_time  # Calculate elapsed time for each image
    total_time += elapsed_time
    # 将每张图像的处理时间添加到image_times列表中，并打印出来。
    image_times.append(elapsed_time)
    print(f"Time taken for image {i + 1}: {elapsed_time} seconds")  # Print the time taken for each image

# Calculate average time per image
if len(image_times) > 0:
    average_time = total_time / len(image_times)
else:
    average_time = 0

print(f"Average time per image: {average_time:.2f} seconds")
print(f"Total time for all images: {total_time:.2f} seconds")

webpage.save()