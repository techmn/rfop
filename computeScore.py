import argparse
import numpy as np
import torch
import torch.utils.data
from torch.autograd import Variable
import pandas as pd
from retrieval_model import RFOP


ver = 'v3'
heard_lang = 'English'


if (
    (ver == 'v1' and heard_lang not in ['English', 'Urdu']) or
    (ver == 'v3' and heard_lang not in ['English', 'German'])
):
    raise ValueError(f"Invalid combination: ver={ver} and heard_lang={heard_lang}")


assert ver in ['v1', 'v2', 'v3'], f"Invalid value for ver: {ver}"
assert heard_lang in ['English', 'Urdu', 'German'], f"Invalid value for heard_lang: {heard_lang}"

if ver == 'v1':
    assert heard_lang in ['English', 'Urdu'], f"Invalid combination: v1 can't be paired with {heard_lang}"
    unheard_lang = 'Urdu' if heard_lang == 'English' else 'English'

elif ver == 'v3':
    assert heard_lang in ['English', 'German'], f"Invalid combination: v3 can't be paired with {heard_lang}"
    unheard_lang = 'German' if heard_lang == 'English' else 'English'


print('Heard_Language: %s'%(heard_lang))
print('Unheard Language: %s'%(unheard_lang))

def read_data(ver, test_file_face, test_file_voice):
    print('Reading Test Face')
    face_test = pd.read_csv(test_file_face, header=None)
    print('Reading Test Voice')
    voice_test = pd.read_csv(test_file_voice, header=None)
    
    face_test = np.asarray(face_test)
    face_test = face_test[:, :4096]
    voice_test = np.asarray(voice_test)
    voice_test = voice_test[:, :512]
    
    face_test = torch.from_numpy(face_test).float()
    voice_test = torch.from_numpy(voice_test).float()
    return face_test, voice_test

def test(face_test_heard, voice_test_heard, face_test_unheard, voice_test_unheard):
    
    n_class = 64 if ver == 'v1' else 78 if ver == 'v2' else 50
    model = RFOP(FLAGS, face_test_heard.shape[1], voice_test_heard.shape[1], n_class)

    checkpoint = torch.load(FLAGS.ckpt)
    #model.load_state_dict(checkpoint['state_dict'])
    model.load_state_dict(checkpoint)
    print("=> loaded checkpoint '{}'".format('checkpoint.pth.tar'))
    model.eval()
    model.cuda()
    
    if FLAGS.cuda:
        face_test_heard, voice_test_heard = face_test_heard.cuda(), voice_test_heard.cuda()
        face_test_unheard, voice_test_unheard = face_test_unheard.cuda(), voice_test_unheard.cuda()

    face_test_heard, voice_test_heard = Variable(face_test_heard), Variable(voice_test_heard)
    face_test_unheard, voice_test_unheard = Variable(face_test_unheard), Variable(voice_test_unheard)

    print('Computing scores')
    with torch.no_grad():
        _, face_heard, voice_heard, _, _  = model(face_test_heard, voice_test_heard)
        _, face_unheard, voice_unheard, _, _  = model(face_test_unheard, voice_test_unheard)
                
        face_heard, voice_heard = face_heard.data, voice_heard.data
        face_unheard, voice_unheard = face_unheard.data, voice_unheard.data
        
        face_heard, voice_heard = face_heard.cpu().detach().numpy(), voice_heard.cpu().detach().numpy()
        face_unheard, voice_unheard = face_unheard.cpu().detach().numpy(), voice_unheard.cpu().detach().numpy()
        
        scores_heard = np.linalg.norm(face_heard - voice_heard, axis=1, keepdims=True)
        scores_unheard = np.linalg.norm(face_unheard - voice_unheard, axis=1, keepdims=True)
        
        print('Writing scores to file')
        
        keys_heard = []
        keys_unheard = []
        with open('../%s/%s_test.txt'%(ver, heard_lang), 'r+') as f:
            for dat in f:
                keys_heard.append(dat.split(' ')[0])
                
        with open('../%s/%s_test.txt'%(ver, unheard_lang), 'r+') as f:
            for dat in f:
                keys_unheard.append(dat.split(' ')[0])


        with open('./sub_score_%s_heard.txt'%(heard_lang), 'w') as f:
            for i, dat in enumerate(scores_heard):
                f.write('%s %f\n'%(keys_heard[i], dat))
                
        with open('./sub_score_%s_unheard.txt'%(unheard_lang), 'w') as f:
            for i, dat in enumerate(scores_unheard):
                f.write('%s %f\n'%(keys_unheard[i], dat))
        
    return 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--cuda', action='store_true', default=True, help='CUDA training')
    parser.add_argument('--ckpt', type=str, default='./checkpoint_english_train.pth.tar', help='Checkpoints directory.')    
    parser.add_argument('--dim_embed', type=int, default=256, help='Embedding Size')
    
    global FLAGS
    FLAGS, unparsed = parser.parse_known_args()
    FLAGS.cuda = torch.cuda.is_available()
    torch.manual_seed(FLAGS.seed)

    if FLAGS.cuda:
        torch.cuda.manual_seed(FLAGS.seed)

    print('Loading Heard Language Data')
    test_file_face = '../%s/%s/%s_faces_test.csv'%(ver, heard_lang, heard_lang)
    test_file_voice = '../%s/%s/%s_voices_test.csv'%(ver, heard_lang, heard_lang)
    face_test_heard, voice_test_heard = read_data(ver, test_file_face, test_file_voice)

    print('Loading UnHeard Language Data')
    test_file_face = '../%s/%s/%s_faces_unheard_test.csv'%(ver, heard_lang, unheard_lang)
    test_file_voice = '../%s/%s/%s_voices_unheard_test.csv'%(ver, heard_lang, unheard_lang)
    face_test_unheard, voice_test_unheard = read_data(ver, test_file_face, test_file_voice)

    test(face_test_heard, voice_test_heard, face_test_unheard, voice_test_unheard)
