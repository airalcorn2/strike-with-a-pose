import pkg_resources
from PyQt5 import QtCore
from PyQt5.QtWidgets import QLabel, QPushButton
import pickle
import os
from torchvision import transforms
from PIL import Image
import urllib
import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence

VOCAB_PATH = pkg_resources.resource_filename("strike_with_a_pose", "data/vocab.pkl")
EMBED_SIZE = 256    # dimension of word embedding vectors
HIDDEN_SIZE = 512   # dimension of lstm hidden states'
NUM_LSTM_LAYER = 1  # number of layers in lstm
ENCODER_PATH = pkg_resources.resource_filename("strike_with_a_pose", "models/encoder-5-3000.pkl")
DECODER_PATH = pkg_resources.resource_filename("strike_with_a_pose", "models/decoder-5-3000.pkl")

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")




class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(EncoderCNN, self).__init__()
        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-1]  # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

    def forward(self, images):
        """Extract feature vectors from input images."""
        with torch.no_grad():
            features = self.resnet(images)
        features = features.reshape(features.size(0), -1)
        features = self.bn(self.linear(features))
        return features


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, max_seq_length=20):
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.max_seg_length = max_seq_length

    def forward(self, features, captions, lengths):
        """Decode image feature vectors and generates captions."""
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True)
        hiddens, _ = self.lstm(packed)
        outputs = self.linear(hiddens[0])
        return outputs

    def sample(self, features, states=None):
        """Generate captions for given image features using greedy search."""
        sampled_ids = []
        inputs = features.unsqueeze(1)
        for i in range(self.max_seg_length):
            hiddens, states = self.lstm(inputs, states)  # hiddens: (batch_size, 1, hidden_size)
            outputs = self.linear(hiddens.squeeze(1))  # outputs:  (batch_size, vocab_size)
            _, predicted = outputs.max(1)  # predicted: (batch_size)
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)  # inputs: (batch_size, embed_size)
            inputs = inputs.unsqueeze(1)  # inputs: (batch_size, 1, embed_size)
        sampled_ids = torch.stack(sampled_ids, 1)  # sampled_ids: (batch_size, max_seq_length)
        return sampled_ids

class ImgCaptioner:
    def __init__(self):

        # we use the pre-trained the models from
        # https://www.dropbox.com/s/ne0ixz5d58ccbbz/pretrained_model.zip?dl=0
        if not os.path.isfile(ENCODER_PATH):
            print("Downloading the encoder pkl...")
            url = "http://auburn.edu/~qzl0019/share/encoder_5_3000.pkl"
            urllib.request.urlretrieve(url, ENCODER_PATH)
        if not os.path.isfile(DECODER_PATH):
            print("Downloading the decoder pkl...")
            url = "http://auburn.edu/~qzl0019/share/decoder_5_3000.pkl"
            urllib.request.urlretrieve(url, DECODER_PATH)

        # Load vocabulary wrapper
        with open(VOCAB_PATH, 'rb') as f:
            self.vocab = pickle.load(f)

        # Build models
        encoder = EncoderCNN(EMBED_SIZE).eval()  # eval mode (batchnorm uses moving mean/variance)
        decoder = DecoderRNN(EMBED_SIZE, HIDDEN_SIZE, len(self.vocab), NUM_LSTM_LAYER)
        self.encoder = encoder.to(DEVICE)
        self.decoder = decoder.to(DEVICE)

        # Load the trained model parameters
        self.encoder.load_state_dict(torch.load(ENCODER_PATH))
        self.decoder.load_state_dict(torch.load(DECODER_PATH))


    @staticmethod
    def get_gui_comps():

        # Prediction text.
        img_caption = QLabel("<strong>ImageCaption</strong>: <br>")
        img_caption.setTextInteractionFlags(QtCore.Qt.TextBrowserInteraction)

        # Predict button.
        predict = QPushButton("Captioning")
        # Order matters. Prediction button must be named "predict" in tuple.
        return [("img_caption", img_caption), ("predict", predict)]

    def init_scene_comps(self):
        pass

    def forward(self, input_image):

        # Image preprocessing
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))])
        # Prepare an image
        image = input_image.resize([224, 224], Image.LANCZOS)
        image = transform(image).unsqueeze(0)
        image_tensor = image.to(DEVICE)

        # Generate an caption from the image
        feature = self.encoder(image_tensor)
        sampled_ids = self.decoder.sample(feature)
        sampled_ids = sampled_ids[0].cpu().numpy()  # (1, max_seq_length) -> (max_seq_length)
        return sampled_ids

    def predict(self, image):
        with torch.no_grad():

            sampled_ids = self.forward(image)

            # Convert word_ids to words
            sampled_caption = []
            for word_id in sampled_ids:
                word = self.vocab.idx2word[word_id]
                sampled_caption.append(word)
                if word == '<end>':
                    break
            sentence = ' '.join(sampled_caption)

            self.img_caption.setText("<strong>ImageCaption</strong>:<br>"
                                   "{0}<br>".format(sentence))

    def render(self):
        pass

    def clear(self):
        pass
