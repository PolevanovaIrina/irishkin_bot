import os
import shutil

from aiogram import Bot, types
from aiogram.dispatcher import Dispatcher
from aiogram.utils import executor


from aiogram.types import ReplyKeyboardRemove, \
    ReplyKeyboardMarkup, KeyboardButton, \
    InlineKeyboardMarkup, InlineKeyboardButton

from PIL import Image
import numpy as np
from people_segmentation.pre_trained_models import create_model

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision.transforms as transforms
import torchvision.models as models

import copy

model_seg = create_model("Unet_2020-07-20")
model_seg.eval()

imsize_1 = 256

loader = transforms.Compose([
    transforms.Resize(imsize_1),
    transforms.CenterCrop(imsize_1),
    transforms.ToTensor()])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def image_loader(image_name, i ):
    if i ==1:
      image = Image.open(image_name)
      image = loader(image).unsqueeze(0)

    return image.to(device, torch.float)

class ContentLoss(nn.Module):

    def __init__(self, target, ):
        super(ContentLoss, self).__init__()
        self.target = target.detach()
        self.loss = F.mse_loss(self.target, self.target)

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input

class StyleLoss_1(nn.Module):

    def __init__(self, target_feature, mask, i, style_weight):
        super(StyleLoss_1, self).__init__()
        if i == 0:
            self.i = 0
            self.style_weight = style_weight
            self.target = self.gram_matrix(target_feature).detach()
        else:
            self.i = 1
            self.one = torch.ones_like(mask)
            self.mask = torch.addcmul(self.one, self.one, mask, value=-1)
            self.style_weight = style_weight
            self.mask = torch.cat(target_feature.size()[1] * [self.mask.unsqueeze(0)]).unsqueeze(0).detach()
            self.target = self.gram_matrix(target_feature * self.mask).detach()


    def gram_matrix(self, input):
        batch_size, f_map_num, h, w = input.size()
        features = input.view(batch_size * h, w * f_map_num)
        G = torch.mm(features, features.t())
        G = G.div(batch_size * h * w * f_map_num)
        return G

    def forward(self, input):
        if self.i == 0:
            G = self.gram_matrix(input)
            self.loss = F.mse_loss(G, self.target)
            self.loss = self.loss * self.style_weight
        else:
            G = self.gram_matrix(input * self.mask)
            self.loss = F.mse_loss(G, self.target)
            self.loss = self.loss * self.style_weight
        return input

class Normalization(nn.Module):

    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std

cnn = models.vgg19_bn(pretrained=True).features.to(device).eval()

class Style_Transfer:

    def __init__(self, style_img_1, content_img, background, model_seg, cnn, num_steps=8):

        self.cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
        self.cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)
        self.style_weights = [0.75, 0.5, 0.2, 0.2, 0.2]
        self.content_layers = ['conv_4']
        self.style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
        self.cnn = cnn
        self.style_img_1 = style_img_1
        self.content_img = content_img
        self.background = background
        self.input_img = content_img.clone()
        self.model_seg = model_seg
        self.num_steps = num_steps
        self.style_weight_1 = 100000000
        self.style_loss = None

    def gram_matrix(self, input):
        batch_size, f_map_num, h, w = input.size()
        features = input.view(batch_size * h, w * f_map_num)
        G = torch.mm(features, features.t())
        G = G.div(batch_size * h * w * f_map_num)
        return G

    def replace_layers(self):
        for i, layer in enumerate(self.cnn):
            if isinstance(layer, torch.nn.MaxPool2d):  # заменим MaxPool2d на AvgPool2d, так стиль переносится лучше
                self.cnn[i] = torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

    def get_style_model_and_losses(self, mask):
        self.cnn = copy.deepcopy(self.cnn)
        self.replace_layers()
        normalization = Normalization(self.cnn_normalization_mean, self.cnn_normalization_std).to(device)

        content_losses = []
        style_losses = []

        model = nn.Sequential(normalization)

        i = 0
        for layer in self.cnn.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = 'conv_{}'.format(i)
            elif isinstance(layer, nn.ReLU):
                name = 'relu_{}'.format(i)
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.AvgPool2d):
                name = 'pool_{}'.format(i)
                mask_layer = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
                mask = mask.unsqueeze(0)
                mask = mask_layer(mask)  # применяем слой на маску, чтобы согласовывать размеры
                mask = mask.squeeze(0)

            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(i)
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

            model.add_module(name, layer)

            if name in self.content_layers:
                target = model(self.content_img).detach()
                content_loss = ContentLoss(target)
                model.add_module("content_loss_{}".format(i), content_loss)
                content_losses.append(content_loss)

            if name in self.style_layers:
                target_feature_1 = model(self.style_img_1).detach()
                style_loss = StyleLoss_1(target_feature_1, mask, self.background, self.style_weights[i - 1])
                model.add_module("style_loss_1{}".format(i), style_loss)
                style_losses.append(style_loss)

        for i in range(len(model) - 1, -1, -1):
            if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss_1):
                break

        model = model[:(i + 1)]

        return model, style_losses, content_losses

    def get_input_optimizer(self):
        optimizer = optim.LBFGS([self.input_img.requires_grad_()])
        return optimizer

    def run_style_transfer(self):
        """Run the style transfer."""
        print('Building the style transfer model..')
        image = self.content_img
        with torch.no_grad():
            prediction = self.model_seg(image)[0][0]
        mask = (prediction > 0).cpu().numpy().astype(np.uint8)
        mask = torch.from_numpy(mask)
        mask = mask.to(dtype=torch.float32)
        model, style_losses, content_losses = self.get_style_model_and_losses(mask)
        optimizer = self.get_input_optimizer()

        print('Optimizing..')
        run = [0]
        while run[0] <= self.num_steps:

            def closure():

                self.input_img.data.clamp_(0, 1)

                optimizer.zero_grad()

                model(self.input_img)

                style_score_1 = 0
                style_score_2 = 0
                content_score = 0

                for sl in style_losses:
                    style_score_1 += sl.loss
                for cl in content_losses:
                    content_score += cl.loss

                # взвешивание ощибки
                style_score_1 *= 10000000
                content_score *= 1

                loss = style_score_1 + content_score
                loss.backward()

                run[0] += 1
                if run[0] % 50 == 0:
                    print("run {}:".format(run))
                    print('Style Loss 1 : {:4f} Content Loss: {:4f}'.format(
                        style_score_1.item(), content_score.item()))
                    print()

                return style_score_1 + content_score

            optimizer.step(closure)

        # a last correction...
        self.input_img.data.clamp_(0, 1)

        return self.input_img


TOKEN = '1470750612:AAHyjk4QmXufqgzjR9kdTpc0WcDmHHnEyXY'

bot = Bot(token=TOKEN)
dp = Dispatcher(bot)

button_hi = KeyboardButton('Привет! 👋')

greet_kb = ReplyKeyboardMarkup(resize_keyboard=True)
greet_kb.add(button_hi)

users_photo = {}
users_type = {}
test_counter = 0

inline_kb_full = InlineKeyboardMarkup(row_width=3)
inline_kb_full.add(InlineKeyboardButton('Сделай из лета зиму', callback_data='btn2'))
inline_kb_full.add(InlineKeyboardButton('Перенеси стиль на фон за мной', callback_data='btn1'))
inline_kb_full.add(InlineKeyboardButton('Перенеси стиль на всё фото', callback_data='btn3'))

@dp.callback_query_handler(lambda c: c.data == 'btn1')
async def process_callback_button1(callback_query: types.CallbackQuery):
    global users_photo
    await bot.answer_callback_query(callback_query.id)
    users_photo.update({callback_query.from_user.id: []})
    users_type.update({callback_query.from_user.id: 'Background_style'})
    await bot.send_message(callback_query.from_user.id, 'Загрузите фото стиля')

@dp.callback_query_handler(lambda c: c.data == 'btn3')
async def process_callback_button1(callback_query: types.CallbackQuery):
    global users_photo
    await bot.answer_callback_query(callback_query.id)
    users_photo.update({callback_query.from_user.id: []})
    users_type.update({callback_query.from_user.id: 'Change_style'})
    await bot.send_message(callback_query.from_user.id, 'Загрузите фото стиля')


@dp.callback_query_handler(lambda c: c.data == 'btn2')
async def process_callback_button1(callback_query: types.CallbackQuery):
    global users_photo
    await bot.answer_callback_query(callback_query.id)
    await bot.send_message(callback_query.from_user.id, 'Загрузите фото')
    users_photo.update({callback_query.from_user.id: []})
    users_type.update({callback_query.from_user.id: 'WinterSummer'})
    print('new user', users_photo)

@dp.message_handler(commands=['start'])
async def process_start_command(message: types.Message):
    await bot.send_message(message.from_user.id, "Привет, я бот, который может 2 вещи:  Перенесли стиль с одной фотографии на другую, а так же сделать из летнего пейзажа зимний и наоборот", reply_markup=greet_kb)
    await bot.send_message(message.from_user.id, "Чтобы сказать мне что сделать, нажни на клавиатуре кнопку 'Привет! 👋", reply_markup=inline_kb_full)


@dp.message_handler(commands=['help'])
async def process_help_command(message: types.Message):
    await bot.send_message(message.from_user.id,"Нажми на кнопку 'Привет! 👋' и я обработаю фото для тебя!")


@dp.message_handler()
async def echo_message(message: types.Message):
    global test_counter
    test_counter += 1
    if message.text.lower() == 'Привет! 👋':
        await message.reply('Привет! 👋', reply_markup=inline_kb_full)
    else:
        await message.reply(f"Я не совсем понимаю тебя,нажми кнопку 'Привет! 👋' и выбери, что я могу для тебя сделать ", reply_markup=inline_kb_full)


@dp.message_handler(content_types=['photo'])
async def image_handler(message: types.Message):
    global users_photo
    global background
    user_id = message.from_user.id

    input_dir = f'./input_images/{user_id}'
    input_dir_nn = f'../input_images/{user_id}'
    output_dir = f'./output_images/{user_id}'
    output_dir_nn = f'../output_images/{user_id}'

    if users_type.get(user_id) is None:
        await message.reply('Я не понимаю, что мне с этим делать, выберите, пожалуйста', reply_markup=inline_kb_full)
    else:
        if os.path.isdir(input_dir) == False:
            os.mkdir(input_dir)
        if users_type.get(user_id) == 'Background_style':
            file = await bot.get_file(message.photo[-1].file_id)
            await message.photo[-1].download(f'{input_dir}/style_image.jpg')
            photos = users_photo.get(user_id, [])
            photos.append(file)
            users_photo[user_id] = photos
            users_type.update({user_id: 'Background_content'})
            background = 1
            await bot.send_message(message.from_user.id, 'Загрузите фото контента')

        elif users_type.get(user_id) == 'Change_style':
            file = await bot.get_file(message.photo[-1].file_id)
            await message.photo[-1].download(f'{input_dir}/style_image.jpg')
            # print(users_photo.get(user_id))
            photos = users_photo.get(user_id, [])
            photos.append(file)
            # print(users_photo)
            users_photo[user_id] = photos
            users_type.update({user_id: 'Background_content'})
            background = 0
            await bot.send_message(message.from_user.id, 'Загрузите фото контента')



        elif users_type.get(user_id) == 'Background_content':
            # todo: тут падало без await
            file = await bot.get_file(message.photo[-1].file_id)
            await message.photo[-1].download(f'{input_dir}/content_image.jpg')
            photos = users_photo.get(user_id, [])
            photos.append(file)
            users_photo[user_id] = photos
            await bot.send_message(message.from_user.id, 'Чуть-чуть магии *думою*')
            content_image = image_loader(f'{input_dir}/content_image.jpg', 1)
            style_image = image_loader(f'{input_dir}/style_image.jpg', 1)
            style_transfer = Style_Transfer(style_image, content_image, background, model_seg, cnn)
            output = style_transfer.run_style_transfer()
            output = transforms.ToPILImage()(output[0])
            output.save(f'{input_dir}/content_image.jpg')
            file = types.InputFile(f'{input_dir}/content_image.jpg')

            await bot.send_photo(user_id, photo=file)
            await bot.send_message(message.from_user.id, 'Держи')
            if os.path.isdir(input_dir):
                shutil.rmtree(input_dir)
            if os.path.isdir(output_dir):
                shutil.rmtree(output_dir)



        else:

            try:
                os.mkdir(input_dir)

                await message.photo[-1].download(f'{input_dir}/image.jpg')

                img = image_loader(f'{input_dir}/image.jpg', 1)
                img = transforms.ToPILImage()(img[0])
                img.save(f'{input_dir}/image.jpg')
                img = image_loader(f'{input_dir}/image.jpg', 1)

                if os.system(f'cd pytorch-CycleGAN-and-pix2pix && '
                             f'test.py --dataroot {input_dir_nn} '
                             f'--name summer2winter_yosemite_pretrained '
                             f'--model test --no_dropout --gpu_ids -1 '
                             f'--results_dir {output_dir_nn}') != 0:
                    raise Exception('Something in neural network went wrong!')

                with torch.no_grad():
                    prediction = model_seg(img)[0][0]
                mask = (prediction > 0).cpu().numpy().astype(np.uint8)
                mask = torch.from_numpy(mask)
                mask = mask.to(dtype=torch.float32)

                one = torch.ones_like(mask)
                img_after = image_loader(f'{output_dir}/summer2winter_yosemite_pretrained/test_latest/images/image_fake.png', 1)
                img_after = torch.addcmul(img_after, img_after, mask, value=-1)
                mask = torch.addcmul(one, one, mask, value=-1)
                mask = torch.cat(img.size()[1] * [mask.unsqueeze(0)]).unsqueeze(0).detach()
                img = torch.addcmul(img, img, mask, value=-1)
                img_after = torch.addcmul(img_after, img, one, value=1)
                img_fin = transforms.ToPILImage()(img_after[0])
                img_fin.save(f'{output_dir}/summer2winter_yosemite_pretrained/test_latest/images/image_fake.png')

                file = types.InputFile(
                    f'{output_dir}/summer2winter_yosemite_pretrained/test_latest/images/image_fake.png')


                await bot.send_photo(user_id, photo=file)
            finally:
                if os.path.isdir(input_dir):
                    shutil.rmtree(input_dir)
                if os.path.isdir(output_dir):
                    shutil.rmtree(output_dir)


if __name__ == '__main__':
    shutil.rmtree('../input_images')
    os.mkdir('../input_images')

    executor.start_polling(dp)
