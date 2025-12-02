import os
import pathlib
from MCTSv2.task import MCTS_Task
import argparse
from tqdm import tqdm
import networks as nx
import pyarrow.parquet as pq
from datasets import load_dataset
from utils.tools import *
from MCTSv2.base import*
from datetime import datetime
from models.inference_models import *
import traceback
from sentence_transformers import SentenceTransformer
from similarities import *

MODEL_PATH = {'qwen7b':'/workspace/LLaMA-Factory/models/Qwen2___5-7B-Instruct',
              'llama3':'/workspace/LLaMA-Factory/models/Meta-Llama-3-8B-Instruct',
              'llama3.1':'/workspace/LLaMA-Factory/models/Llama3-1-8B-Instruct',
              'qwen14b':'/workspace/LLaMA-Factory/models/Qwen2.5-14B-Instruct',
              'qwenqwq':'/workspace/LLaMA-Factory/models/QwQ-32B-Preview',
              'qwen32b':'/workspace/LLaMA-Factory/models/Qwen2.5-32B-Instruct'}

gte_model_path = '/workspace/LLaMA-Factory/models/gte_Qwen2-7B-instruct'
emb_model_path = '/workspace/LLaMA-Factory/models/text2vec-base-multilingual'
tog_model_path = '/workspace/LLaMA-Factory/models/msmarco-distilbert-base-tas-b'

def run(arguments:argparse.ArgumentParser):
    print('-'*30, 'Begin inference', '-'*30, '\n')
    if arguments.use_freebase: # freebase type
        input_dir = f'/workspace/longxiao/KGQA/ToG-main/data/{arguments.task_name}.json' # freebase type
        try:
            dataset, question_string, q_string = prepare_dataset(input_dir) # freebase type
            data_len = len(dataset)
            # pdb.set_trace()
        except Exception as e:
            print(f'File must be standardized json!\nError type:{e}\n')
            return
    else: # graph type
        input_dir = f'/workspace/longxiao/KGQA/MCTS-KGQA/data/KGData/{arguments.task_name}/'
        try:
            dataset = read_data(input_dir, mode='test') # graph type
            data_len = len(dataset)
            # pdb.set_trace()
        except Exception as e:
            print(f'File must be standardized json!\nError type:{e}\n')
            return

    assert data_len > 0, "Data list is empty!\n"
    # assert 'content' in data_list[0].keys() and 'answer' in data_list[0].keys(), "Key error, Make sure json object contain correct keys!\n"
    output_list = []
    correct_count = 0
    path_list = []
    tree_list = []
    now = datetime.now() # 获取当前的小时和分钟
    current_day, current_month, current_hour, current_minute, current_second = now.day, now.month, now.hour, now.minute, now.second
    print(f"CUDA_VISIBLE_DEVICES set to: {os.environ['CUDA_VISIBLE_DEVICES']}")
    
    # gte_model = SentenceTransformer(gte_model_path)
    # text2vec_model = BertSimilarity(model_name_or_path=emb_model_path)
    # sentence_bert = SentenceTransformer(tog_model_path)
    
    # if arguments.emb_model == 'gte':
    #     emb_model = gte_model.to(torch.device("cpu"))
    # elif arguments.emb_model == 'text2vec':
    #     emb_model = text2vec_model.to(torch.device("cpu"))
    # elif arguments.emb_model == 'sentence_bert':
    #     emb_model = sentence_bert.to(torch.device("cpu"))
    
    tokenizer, model = None, None
    if arguments.use_local_method:
        if arguments.use_vllm: # 这里是为了多次采样 投票决定最有可能的关系
            from models.vllm_models import load_vLLM_model
            if arguments.propose_method == 'qwen':
                model_path = MODEL_PATH['qwen']
                tokenizer, model = load_vLLM_model(model_path)
                print("*"*50, "加载qwen模型！", "*"*50)
            if arguments.propose_method == 'llama':
                model_path = MODEL_PATH['llama']
                tokenizer, model = load_vLLM_model(model_path)
                print("*"*50, "加载llama模型！", "*"*50)
        else: #不使用vllm 
            if 'qwen' in arguments.propose_method:
                model_path = MODEL_PATH[arguments.propose_method]
                from models.inference_models import get_inference_model_qwen
                tokenizer, model = get_inference_model_qwen(model_path)
                print("*"*50, "加载qwen模型！", "*"*50)              
            elif 'llama' in arguments.propose_method:
                model_path = MODEL_PATH[arguments.propose_method]
                from models.inference_models import get_inference_model_llama
                tokenizer, model = get_inference_model_llama(model_path)
                print("*"*50, "加载llama模型！", "*"*50)
    else: # 使用api 不需要加载模型
        pass
    # poor_case = ["WebQTest-12", "WebQTest-20", "WebQTest-39", "WebQTest-44", "WebQTest-58", "WebQTest-62", 'WebQTest-77', 'WebQTest-115', 'WebQTest-180', 'WebQTest-182', 'WebQTest-221', 'WebQTest-231', 'WebQTest-257', 'WebQTest-261', 'WebQTest-277', 'WebQTest-367', 'WebQTest-375', 'WebQTest-421', 'WebQTest-432', 'WebQTest-450', 'WebQTest-472', 'WebQTest-523', 'WebQTest-535', 'WebQTest-549', 'WebQTest-564', 'WebQTest-590', 'WebQTest-608', 'WebQTest-609', 'WebQTest-656', 'WebQTest-671', 'WebQTest-676', 'WebQTest-689', 'WebQTest-696', 'WebQTest-708', 'WebQTest-728', 'WebQTest-734', 'WebQTest-759', 'WebQTest-778', 'WebQTest-785', 'WebQTest-836', 'WebQTest-868', 'WebQTest-873', 'WebQTest-884', 'WebQTest-918', 'WebQTest-936', 'WebQTest-941', 'WebQTest-943', 'WebQTest-944', 'WebQTest-1052', 'WebQTest-1118', 'WebQTest-1143', 'WebQTest-1145', 'WebQTest-1154', 'WebQTest-1179', 'WebQTest-1200', 'WebQTest-1203', 'WebQTest-1240', 'WebQTest-1252', 'WebQTest-1271', 'WebQTest-1279', 'WebQTest-1296', 'WebQTest-1350', 'WebQTest-1388', 'WebQTest-1436', 'WebQTest-1443', 'WebQTest-1468', 'WebQTest-1477', 'WebQTest-1478', 'WebQTest-1480', 'WebQTest-1523', 'WebQTest-1539', 'WebQTest-1544', 'WebQTest-1547', 'WebQTest-1554', 'WebQTest-1558', 'WebQTest-1563', 'WebQTest-1568', 'WebQTest-1597', 'WebQTest-1622', 'WebQTest-1707', 'WebQTest-1759', 'WebQTest-1774', 'WebQTest-1808', 'WebQTest-1811', 'WebQTest-1853', 'WebQTest-1884', 'WebQTest-1938', 'WebQTest-1955', 'WebQTest-1960', 'WebQTest-1969', 'WebQTest-1975', 'WebQTest-1996', 'WebQTest-2013', 'WebQTest-2019']
    poor_case = []
    poor_cwq_p1 = ['WebQTest-590_6aad73acb74f304bc9acae44314164be', 'WebQTrn-662_7a992044f94b39edfc37ac5dcfcb3c26', 'WebQTrn-1864_9dc4e22121d3a46d45b8f9bd9e8c7013', 'WebQTest-12_68d745a0657c86906382873e57294d6a', 'WebQTrn-1864_67ecd1c247c3b2c9545fbcf1ad8d9d00', 'WebQTest-361_e24533e28da40db99eb4b25773f9d38f', 'WebQTrn-810_e3d40457273785e46c5b71732713a5f4', 'WebQTrn-60_39ba4faa87698cb0767d1a5ee7ce1827', 'WebQTrn-846_a29552911617e890ca2e1d6564e0990e', 'WebQTrn-2444_e5059ff268415917df330817b9c8ef8c', 'WebQTest-1812_c29807a955ea46fa79cdc9f7aeacba18', 'WebQTest-590_2f40ab1d4ea497555c4c883c92a758b7', 'WebQTest-1012_7ee41dcd5e0ee726e65a46d891982c8f', 'WebQTrn-2026_84e5ab7013a2c0dd11f7f61db45df94c', 'WebQTrn-60_eae3033643523fd3a54b6c147fdfa728', 'WebQTest-12_c701ad2b5b8ef3f3ed26dd2ed8703d05', 'WebQTrn-2444_06d5101218b4299633a55b1f229e9b40', 'WebQTrn-60_ab97cd852bcb8f41194f686282460b40', 'WebQTrn-3100_84e5ab7013a2c0dd11f7f61db45df94c', 'WebQTrn-3335_a9b18d079f29555cbd1e1740c7d6e40e', 'WebQTest-12_7b54b31f3e5a6273f4fd2a20e565ec6d', 'WebQTrn-710_e3d40457273785e46c5b71732713a5f4', 'WebQTrn-3694_4089d6d7ba86121ff285b80865682a7c', 'WebQTest-361_89e48a343537ccd0148b822f63519c99', 'WebQTrn-105_cf2c2ca737c00135df125459635a368a', 'WebQTrn-1155_fbf8e0d41d59c1c1f57016316d470540', 'WebQTest-361_5f8c05737061d5501b6a23a978b5589b', 'WebQTrn-2314_b274ade1f00c84f7244131327c785b4a', 'WebQTrn-3384_275abff5d62790363fe8bc0387660115', 'WebQTest-100_1619341addc048f21b246e33e2458609', 'WebQTrn-2057_522769c85c8189a870c317c0aa940eaa', 'WebQTest-106_f21253e8b8df81c378590e68b2cd107e', 'WebQTrn-2189_c0495334f4678f947149a7a30d8e49a0', 'WebQTrn-2748_65952c606f7a7204cc1bec237c146bfa', 'WebQTrn-64_08a3071aec88af141fc20ed22cfff0e2', 'WebQTest-1875_de764c65ac3c6a870a407dd9cafd0dc9', 'WebQTrn-60_8b2e8f4e23e7af4e77ae1535c753ef6b', 'WebQTest-1686_554811ebe1463287ee640a214683ea57', 'WebQTrn-14_1d1c28cc9c5f4e83f9f932e3615e2391', 'WebQTrn-2314_f53b7962c4dfca88044f3c0a89ac0290', 'WebQTest-1923_5584d7ea5e4b9f2391d1610bcc5f75fb', 'WebQTrn-662_707caa73f11b1e655a0f5f6b75082734', 'WebQTrn-1817_e3cca7a9117d7c23ddfc189a49770034', 'WebQTest-361_d4b7b091dfd540b6b8efcab6634968d1', 'WebQTest-1306_213cbfafc2612b42c5f8efe85c3532c5', 'WebQTest-106_ac6916d7c5822d5edfe0f67c77f97a15', 'WebQTest-213_024fd6ca0b4cb30927c22e93a552ae6c', 'WebQTest-1384_563b401a13e5a2b9bc3c9ec4bbe58962', 'WebQTrn-3166_762847d07f73b2ca5fd1504d7dcd9d9a', 'WebQTest-1260_c4e06c3a9e4b4f10bd1ae97f1742c198', 'WebQTrn-436_b301288f1064d4357ae5d8271061e1d1', 'WebQTrn-2218_dacbec08f8d31829038751934850ed95', 'WebQTrn-662_5ec0304dbb715858e564cb8426b7e4f7', 'WebQTest-361_c4a62b0d37183dfdd1ca06ad4dfafcf5', 'WebQTrn-1758_82ba7a28dcbd66ab8e128c8975b2e692', 'WebQTest-1513_020f93c678f201a55199a0c40d829467', 'WebQTrn-3087_a82e8e886228ce0d1aecee4a2caab326', 'WebQTrn-2189_50cf52311d0cdd04337ffc0d5378b1b8', 'WebQTrn-2069_a146b4445fb7b259407068b18faf0553', 'WebQTrn-3766_2458404b2726ce40f3d525263dc9f74c', 'WebQTrn-436_d4fb347b7326355a4457aa2ce1d502f3', 'WebQTrn-2189_cf2c2ca737c00135df125459635a368a', 'WebQTest-590_e1cd6a19c1fe109a00e4149625e31531', 'WebQTrn-3087_6f6c0d89afdd0d422980cd8024d5cbd1', 'WebQTrn-567_98d888d83a1b23a19245b02c10839f3f', 'WebQTest-1528_f4911e93c40f00f8c256e3ccf42422c1', 'WebQTest-12_a91de216aa67ef8beef840ad8ad1d1be', 'WebQTrn-2784_abedc8acd0fdcd2e595d7ec6d58d6058', 'WebQTest-191_1ca9a284fcd2c17f453b0fb590e8223f', 'WebQTest-1941_4ccdeac69c947afd15a0b301091498c0', 'WebQTrn-303_c413d25bc8fb85a551d2619016da958b', 'WebQTest-983_fda09a5c3c6128b05dcf0db0b231193b', 'WebQTrn-2784_6ab5c32d090e32dca9db642f612076b8', 'WebQTest-1923_1bc85cf9eab0a0714557180ac27e91a2', 'WebQTrn-62_12c5e056c8b1024d48b0f7f4b2d03b7e', 'WebQTrn-2904_92cd88da247f06a2a19f25b5d45c8d9c', 'WebQTrn-2316_47b067b98845fdb2ebaf2442f5ad1298', 'WebQTrn-3084_73a0a036677106856ef62808aa205b70', 'WebQTest-537_a8a56da657efbba90881cce421ae6962', 'WebQTrn-2784_ed45c0405531af3d1f5b49a57c275835', 'WebQTrn-2152_c9be001aaf5f69c9067ba4b530ca0a93', 'WebQTrn-423_ce9f01df27adbfc73deaa6a14e5aa69c', 'WebQTrn-261_367252aa9c16ba8cff6a3bcc5f98dc73', 'WebQTest-654_6ca8cba7511811830a04cd64a8a4cf77', 'WebQTest-538_e767ac9ea2c1ea9896e9267b30ea9306', 'WebQTrn-2215_2fa90adc92b5095b8e1fd27754180250', 'WebQTest-561_c4965c0149340bcfe2620c82edcf5638', 'WebQTrn-25_788ea75164ce87773e55b0753fd1b9f6', 'WebQTrn-1770_6a7c160ace84e7908302f805739ad06d', 'WebQTest-1923_4e9c47a2fdd065e939fe0da5b782f120', 'WebQTrn-3671_fe491c425ea24501a960040ed645a049', 'WebQTrn-2319_ad8dbdcd7415526de37bbead25d2b3b4', 'WebQTest-1528_ac372aaf6e68f8a1bd4b4c8e75972875', 'WebQTrn-738_75d0bf8a1c2aca91051597cd3bdfb371', 'WebQTrn-810_bb7b757a2fe5dfa7023b6b4fe34ead4c', 'WebQTrn-2026_0c8bcc717d50c92c31ff802641504b43', 'WebQTrn-1053_d1f5861f34b782e38a1c3caf9b076af5', 'WebQTest-538_80a0782bf3bf28bbd2cbd5898d2861c2', 'WebQTrn-2152_1694f4392eb0ff79149337ee3f622d91', 'WebQTest-1569_6f91b128b3ff3663ab0fb24cb4bfd1b2', 'WebQTrn-2859_1d0f95cbca773971acb85244d72bed62', 'WebQTrn-2664_51d761aa070710bb5c4b31c8f0dc9ac7', 'WebQTrn-2218_05b93d057f56c7f623f0bed078a219b3', 'WebQTest-1014_af6b19e16eab1809f29e15c0961f90d5', 'WebQTrn-3335_5831bd7b6badbaf458df00184249bb38', 'WebQTrn-2721_963759f4184e1a7f2fdbeea6f16e017e', 'WebQTrn-1938_1ae1a6be469266690c163ee3f6e0293c', 'WebQTrn-1557_83b3d35e3fcf2d866fb2ab3dc21034c0', 'WebQTest-1797_cd2ffcb3be5bcfdcbc61ea440aca05e3', 'WebQTrn-567_2a2de50d3b65cc5d2c88f54e283a4b8a', 'WebQTest-1965_b7c82cd0420f9f1934037e1625db4685', 'WebQTest-1528_2f6fb6d585b98261a3ae6d2a112a4c91', 'WebQTrn-2664_3be7e74ae2c1722d6fc2a22e84d610b4', 'WebQTrn-1677_e3e49622bef77e700894cb4264a4a42d', 'WebQTest-537_a52d362d7abdae05ae86743bdb72a808', 'WebQTrn-1677_80066cc2617f4e1301adab9b3e951711', 'WebQTest-537_3f765ae8a25eecea63667cdb84f5500a', 'WebQTrn-1405_b98a27e21e904173168eb7517b123e51', 'WebQTrn-2286_56479abdf8de613c8157280a9f1d1d1d', 'WebQTrn-1938_b9fa34f09c15f589df8150b880882013', 'WebQTrn-3170_6bd2ce4aab631c5e72a80da02ce8c1b3', 'WebQTrn-567_a597ad8a4ef31ad09f1765593d4e7fcc', 'WebQTrn-124_0ebb5db67d331752adddbe06bca557fc', 'WebQTrn-2316_72d523c9058d95267091aebebd6b7e62', 'WebQTrn-3049_ae2f0556c44ee8e111f11615ad14f9d8', 'WebQTrn-125_003c6a046c47526d922683f22cf0e983', 'WebQTest-1785_64f7f636c2ecaa51fcf62618fe76820a', 'WebQTest-1785_0c2ef888f7d4db8c11f11a94192905ba', 'WebQTrn-1392_ba1b3d8965af4c1f1127fa5219d1c2df', 'WebQTest-1528_f8024d5f486967b395064aa07aa52d6e']
     
    poor_cwq_p2 = ['WebQTrn-3376_118a81ac3fc08ca8590bf8be836d1be1', 'WebQTrn-2784_b64250ae3c9d6c724133d09dad5593ec', 'WebQTest-1301_24c13551b792ceb6ce65520d29b87c61', 'WebQTest-1817_1bbd9ffd7b604510a3d6751b9f8ef796', 'WebQTest-537_21673ea221c99d3d37931bb684325f49', 'WebQTrn-3384_e83e8190540386b20d8b60496674e1d5', 'WebQTest-537_74f1761f9f9f53428fd2495202e0290f', 'WebQTrn-3412_305f49b65a760af0e6ce8d06f43aa22a', 'WebQTrn-62_bce88015327268bc8f081292e2c80dcd', 'WebQTest-538_e92ac6479096acfc8a23fa7ff3fa1b37', 'WebQTrn-3527_4fc72548c4680b0604576961f2ce4459', 'WebQTest-375_3a9a869538584999b9670edcd9773dab', 'WebQTrn-2023_babe81e436454a7d4877c4ae80975c85', 'WebQTest-712_810a930318378edecdffe4b349276794', 'WebQTrn-1646_0c8bcc717d50c92c31ff802641504b43', 'WebQTest-1171_a8ed8aa8e8eaf11e4b00b81b2ac2ac23', 'WebQTrn-2664_c04548701c89d3af786e8b8cc112af82', 'WebQTrn-3249_a5b4b068155de58edc7e632f9bece371', 'WebQTrn-3766_e30e234d3055e7c20ad20d1562018910', 'WebQTest-561_524b7e7fa855202f18ffd7e8c6bdb6c6', 'WebQTrn-2319_0ab11fcdcb73b6d87813cedc74e5a858', 'WebQTest-1785_a46f6053d6c82eaf3574d4a8853999c3', 'WebQTrn-493_c2e01273b1cae913cfd0b14905db6862', 'WebQTrn-42_479799609c49ac2f363e42bda7221b6f', 'WebQTest-1251_128642d73ca29b533dcf6eb1ae0bfa36', 'WebQTrn-1864_b4a27aa1b03dec67ce6ff38f076f53ff', 'WebQTest-1528_c3a338c3684d073518350521455aaa8f', 'WebQTrn-513_2d731534850e1b99f1a4ce2536597327', 'WebQTest-366_e720f6c51fa9762bbf53cde50a3a8542', 'WebQTest-1923_e11655219d44e3762e0510f2bde1c077', 'WebQTrn-1278_69e2c09bf172a29d062ed5bd2973d2bd', 'WebQTrn-64_025fdfafd914ff922ab8144f527c06ec', 'WebQTest-1923_d21baa85614e41d6f4bd2cde1b90e8ad', 'WebQTrn-484_f978435827c7e85a6632a72eea403599', 'WebQTrn-2859_37bf80e89387cf09fca66eb7915a52b0', 'WebQTest-213_7f3bd401a2fe034b67eb41db6a3801b2', 'WebQTrn-567_44fbd8d749b7a4e1415b0fcafb386222', 'WebQTrn-62_b010a81a436481ecd38f027cf6c859d6', 'WebQTrn-567_44fbd8d749b7a4e1415b0fcafb386222', 'WebQTrn-62_b010a81a436481ecd38f027cf6c859d6', 'WebQTest-12_64c67292548a2872e94c2d2162850e82', 'WebQTrn-2784_06d1c376b588c7a7bb36e30c95e914e7', 'WebQTest-537_f1860c3f3024b6bfa5172ee9dfd3248b', 'WebQTest-1785_e87bdab8ec7e430243035f284919f6c5', 'WebQTest-1785_c98260470806382c10fedc92d19e1e41', 'WebQTrn-3766_eaeb39a0db717338715a044a8175bd9a', 'WebQTest-1528_e4576358163b568a25b6d3037836e483', 'WebQTrn-3249_c2f7c4d755631300e1e557415c49ebe1', 'WebQTest-1528_2a2de50d3b65cc5d2c88f54e283a4b8a', 'WebQTrn-2653_d2311125d5969172b311bdb814c7aae4', 'WebQTrn-25_62939f05e216712e4700e993437a980a', 'WebQTest-537_138389537ea07516c4007202a1b8de61', 'WebQTrn-1677_618357e7755882c7fe69ad20987c2e83', 'WebQTrn-1812_688d69d484eeaffda36f06fbba9fe9fb', 'WebQTrn-1069_e83e8190540386b20d8b60496674e1d5', 'WebQTest-712_afd00ae7f259af2831587a8f5ead5d3e', 'WebQTrn-1484_c4d4e06a9ed122d9ffa605447e7772ac', 'WebQTest-654_eacad512118512366a932ffd59ad4578', 'WebQTest-1528_836cde2e6cac496e51007560e3bba8d2', 'WebQTest-538_7084416ab9f72f1f1f8fc3ce7871ee4a', 'WebQTrn-497_a1739c3fe23ae1c5f93a421061ef5a1d', 'WebQTest-1168_1591d6a0e6cef93c6863edb2eb0e198f', 'WebQTrn-493_6c0bee1445f970769fbdfec9e218fde6', 'WebQTrn-3251_acf68a88c0375f50017818448cac05ac', 'WebQTrn-105_eacad512118512366a932ffd59ad4578', 'WebQTrn-2664_a82da5c3eea7c68f4ebeb7fe9ae20288', 'WebQTrn-2319_ec5586b8afdc20aa27592c03e67e0f4f', 'WebQTrn-750_05337caf83129cca8d39755239932866', 'WebQTrn-25_a610062e405b459ea2d34dc14f20bb05', 'WebQTrn-3694_12efa4cee5acdd0e21df02422bc9cbdd', 'WebQTrn-124_d633246c068bc5955873cb486af7ffc6', 'WebQTrn-1532_750c9d853294e414aecb33082306c54d', 'WebQTest-1965_da82edbcfa689cfcdbb1382bab1bcb03', 'WebQTrn-1677_54d53ee732e6acf5d9649daae920602a', 'WebQTrn-3249_4fddfec17159cdf7caa87c1e3e1f34cd', 'WebQTrn-567_688a395a652a9d8c1ef2c54236c488b9', 'WebQTrn-2314_be3d335c63904299f0103e947254a9f7', 'WebQTest-1941_eaeb39a0db717338715a044a8175bd9a', 'WebQTest-55_2d29c992458630b7e3343cc7bd7f4263', 'WebQTrn-1817_b9ee95e24e0e6221b48124d421dd5648', 'WebQTrn-2209_e092ba5f2adfd34acb1ff2d1629bae8e', 'WebQTrn-1597_51bf00b62c1d2a126e4c5a7544d97fff', 'WebQTrn-3251_1a90de3e56087702d12f5ac656bd7be0', 'WebQTest-537_80c32b072f27820081c103131c1e58c4', 'WebQTrn-846_27a3b20f7d1aebfeea7ca284d80cceb5', 'WebQTest-1705_701858265dc8d32eee3d5fadd974a9f1', 'WebQTrn-2316_a459bc48bed76ae151bace77f3ff774a', 'WebQTrn-2784_3db4e5394a73964976f58989d87836ea', 'WebQTest-1528_f4031bab4675b90da1eff58abc5ecc91', 'WebQTrn-2319_d482e9d52e0f0ee50645cc918c643d6b', 'WebQTest-712_c82a1eba72695beab0bf7376904b0212', 'WebQTrn-64_b09c00219764b4f102d325a00e808259', 'WebQTest-1812_65b5730a361fac314fa12f043413a8c2', 'WebQTest-983_1f8b6ebf20d1119eba090d8d0257bdc5', 'WebQTrn-2047_8f22f6393ba4a5375179567a8c895183', 'WebQTrn-1812_29aa4d58e974f804b2a78b9dc96c62c0', 'WebQTrn-1532_4d4660e6a11db0041accec1e048423d7', 'WebQTrn-1812_09b712193256ecf2e5f87ce70bd17e72', 'WebQTest-1528_46a304cecca0e072121d7dc42befb590', 'WebQTrn-3084_3f03848605c6758ff2230a955cd92d65', 'WebQTest-1823_a8cd75f8533d96e3be293401d63fad4b', 'WebQTest-759_2a6f2d7311c4070f0f98f8526d05907c', 'WebQTest-375_e1454c8b479000f97c25ee372d747347', 'WebQTrn-2815_3a40921ae6042ff93cc654f727b80209', 'WebQTest-1528_c10f21ff1b8944c429bdb86f46cc9196', 'WebQTest-548_6b65a7948e8a5a52b1ed74160c4aecb3', 'WebQTrn-513_8e7e3bbdaeb7f45a9b4ebef0f38b9d90', 'WebQTrn-2189_839fa68cda37e890bb3eed45714fcc43', 'WebQTrn-3769_6e0a31334a35147f868ae514f9eb9ab8', 'WebQTrn-1677_4a5f3d407f54c02f61d0a3d24a0727e1', 'WebQTrn-2286_357b1c4b2fdaaf9bd16c1bcabe173ce5', 'WebQTrn-95_6f91b128b3ff3663ab0fb24cb4bfd1b2', 'WebQTrn-1758_f39b4277a0936929ce17ed6b69952eb4', 'WebQTest-1000_c03787125499031191feea58eae0de36', 'WebQTrn-25_33b7d857b3156faee9cb6c16741d8b6a', 'WebQTest-537_d5dd05a2d98b2d655530e211f59b7d5a', 'WebQTest-1923_d745069235738a857658a6095eb79ab7', 'WebQTest-1000_0adacc5f8d85e317c77fbf2ffa83dae9', 'WebQTrn-3249_077663ab8d6ddd23a988ac47f1c6371d', 'WebQTrn-1405_ced20d844a472e7fafced76fae0a1c7c', 'WebQTest-1785_afeade6c28913208addc4d0685fadb3b', 'WebQTrn-1677_a732537089fb8605a3ca78fbe5627eff', 'WebQTest-1686_2e08825361a88f308bfd8faea2e225f0', 'WebQTrn-513_4afc4c42cf5d16a6b4efff290c4cbe15', 'WebQTest-759_fc3f0847c910de399199128e0e779b25', 'WebQTrn-567_3fc1da87d6688d1e68a71f1721401694', 'WebQTest-537_e5da8cda32fb1aa37028f9f7f7b1d3a8', 'WebQTrn-241_a60904278de9f6a712bc87d1d25753d2', 'WebQTrn-124_2184d51f090b6217ca2d7bab77c25712', 'WebQTrn-3671_3af150ded64a80876fe64c059f63f248']
    
    poor_cwq_p3 = ['WebQTrn-2777_d88f49612c5fbbc35c6a9bf1992abd98', 'WebQTrn-3249_5aa3063566fca0f5ff73b2b811e3b1ab', 'WebQTrn-3744_7aad0e73f9e2443786bc511f137be535', 'WebQTrn-25_ff53089f475b9fdfec05e61e26a3fc3c', 'WebQTest-1797_2fb9e2823ccf35d2103fa8846d6f2ca8', 'WebQTrn-2314_019ac2df6831ea7e866fa674ce4a9fae', 'WebQTrn-423_b06e29db67de86515db5880e88452c1d', 'WebQTest-213_61516138603e1aae7ee58e812cc4a2d1', 'WebQTest-1785_23fbfc07989820c1a44f16de0e5291be', 'WebQTrn-513_b86316858f841b05eef0de7961fb4562', 'WebQTrn-3769_47cc02e7e7f83568581f65b7dd3449dd', 'WebQTest-1012_0d314d2bc1fed4c1b4af3286a9f2d934', 'WebQTest-1923_963f10fbb72072fc718bd8cc3e0ec12d', 'WebQTrn-484_c932aa6e6365aa759f5d0f8da236643c', 'WebQTest-1785_2fd3a482f02db22067954809fe7b222b', 'WebQTrn-567_45a71163e7876e1ca8b7c88e38999705', 'WebQTest-1923_3e5432e80a9f372f1a2a45700663bb79', 'WebQTrn-64_7df82f515bf7947c2bfdd404503b50fd', 'WebQTest-1875_f978435827c7e85a6632a72eea403599', 'WebQTrn-25_db9695c22f0f88f0733e1286c7fffe30', 'WebQTest-537_4346bf5ae9bf50583f07d197cd0c724a', 'WebQTrn-1597_c8ab7eb9865b458244093ddedeb05134', 'WebQTest-1941_a20823840ed586d363d240f035da3434', 'WebQTrn-464_0b1efb954250d417c58d39ccd891b696', 'WebQTrn-2784_4b7058a69c5eb71df36a89837b1e4f60', 'WebQTest-538_fc5392eefef107aa88084b05ddd2e246', 'WebQTest-100_7a6ae21348065f95922affec47f394b7', 'WebQTrn-25_4d68c1f4906c41ef2f0ea4e416e2615b', 'WebQTrn-2784_83848dc79b647efd6072c66a56c17b4b', 'WebQTest-538_4c971bef32fb5777fb8b18ead5a702fe', 'WebQTrn-1677_22c4d321fc60b69e1d966c77d4107a74', 'WebQTrn-25_abd563e9aa3fc7103b5aae5a6d45564a', 'WebQTrn-3766_a20823840ed586d363d240f035da3434', 'WebQTest-1785_8e46718c3fc1361ff1c02b62a853b402', 'WebQTrn-2569_a2cdb0c9a5b7b4a76677fa7477b739b2', 'WebQTrn-568_ae176a507290b2f5342c73d91ecfb246', 'WebQTest-537_2229bda45800254a8ecdea64c577ca0a', 'WebQTrn-567_7dc33c4b385de73d2f8f053cdf6f23bb', 'WebQTrn-567_e4d5f38a9487c935ab4462347b06cf15', 'WebQTrn-3671_16a18a87b07cc1f5deb84208c7deff87', 'WebQTest-743_af8b975343e4098913128e9ac7d04c05', 'WebQTest-537_0b9df216de1dc61e927869e6f9dfac2b', 'WebQTrn-1023_0513ae5213a4f69739cdef9e7c160e2b', 'WebQTest-1513_24c13551b792ceb6ce65520d29b87c61', 'WebQTrn-60_51d761aa070710bb5c4b31c8f0dc9ac7', 'WebQTest-537_c8e0c5670b9fa714ebab6aaeb0205736', 'WebQTest-538_22837422e25cf0a05cb0efba9aa80402', 'WebQTest-1785_8bad0a0cbda8b0e72f6a60d6917b95eb', 'WebQTrn-846_b2db88607e6e187c67839c81f66e708f', 'WebQTrn-1405_209a925075130582f176bb9a651b1279', 'WebQTrn-3766_6ed17cf93677d92f2fcd2df541fa96b3', 'WebQTest-1012_8602f4a22371baa72ab6965c0f6625d1', 'WebQTrn-1677_3e55a7ab68be573ccb45f5858cd45196', 'WebQTrn-1677_a7b0673d818a426f402b6a2c2573d137', 'WebQTrn-2968_16d7795d856ce567c83173691845c6bf', 'WebQTrn-1155_4ec9be4c828025d788bc5fb5ebca9e6e', 'WebQTrn-1938_21d93e83b78eb4bae89800d09deafa2f', 'WebQTest-537_1382c112d75f45f0613dc151a28b3f62', 'WebQTrn-1677_9e5f2a9a1412133b69d6d0bffba4310a', 'WebQTest-759_b9e4b613dd3d82a718a49860d45d6a63', 'WebQTrn-1770_fee80219bb52a93bbd48e877ed0adc5d', 'WebQTrn-2653_f36fbb8ee04a37d70b8360ac90a6ef5b', 'WebQTest-832_1afd5634cb719e3b77b4345510599ca8', 'WebQTrn-849_cb535b26088aa7a8cf49a7aba65ef3d3', 'WebQTrn-1394_09b1341b08473e166ce0d9d32edbb1df', 'WebQTrn-2615_ec18b1d1eb4302577d2aafe49881c4e3', 'WebQTest-106_87ddee8816a0dd02511eb7670904d500', 'WebQTrn-846_6463b5d4d186708788ca7b34de172f89', 'WebQTrn-2784_ca84a54e3994231e53ce404b55fafff3', 'WebQTrn-1278_b75f55806dcb1e062aeaaeefe0c1551c', 'WebQTrn-2006_9ad6769cb97331945bf8dfe52958126b', 'WebQTrn-1597_9f8b39da7c3f621417a7bb2a708495b3', 'WebQTrn-60_c85d20633dd493d5f7dfd6f8034d5c95', 'WebQTrn-2209_beb5507aba5b006b3b2e0598989448c8', 'WebQTest-1875_5f93548592ef4987720d1e3f3fe6b3d7', 'WebQTrn-3249_af94010c3bfc3a584ffb01e553dd2e8f', 'WebQTrn-710_c1d5c9c777c291a80c29c70ec43c45ad', 'WebQTest-759_88081dc1a259cbd172b89242384ee1cc', 'WebQTrn-3049_bbb9447a02dcb852802ed6cd4bfa1584', 'WebQTest-484_3e119c02d9c8e58387951defcdfb93fd', 'WebQTrn-3166_adbac8a6e3875142a0d07e9d593ef18c', 'WebQTrn-1677_2fe630fbb46b32aa9774a9417e843503', 'WebQTrn-2286_97d590dcc464128f00db773e0f4ec1ac', 'WebQTest-114_05a60a5e9a6d52b6f23993c4868c111f', 'WebQTrn-1297_02ed5ce4f0d0e6c1b8d15752a1be80ad', 'WebQTrn-567_25a9bf629b27a4c525df016451140b28', 'WebQTest-1785_a856a27ccacbf5c2abe9b1e3569998d1', 'WebQTrn-1770_7c271dcbcf911ac67fff291e0c0a0ee8', 'WebQTrn-2784_449d83fa3257a5bbb8bca7337754ff04', 'WebQTrn-662_b87d6f4be9902915cf4ce73d259aa48c', 'WebQTrn-25_62c571c0f7c5f645e7179596b4a308d2', 'WebQTest-361_d8fb92e9cd8d21e59eb5a6bcacc7dd7e', 'WebQTrn-60_c65c81ea2e26d0f6fbc9c5b0ed754ac7', 'WebQTrn-1394_280498958193053b63b236ec3155cecc', 'WebQTrn-1394_6eb4909a184ab0aac83d44471fb5bd49', 'WebQTest-1306_71c93aa9a352e8738c065f3887d01c12', 'WebQTest-1171_36be0686f0f845782d8e8d0c09704e45', 'WebQTrn-1677_8c2eec7cd5ec451bec6d47d09e8d9c63', 'WebQTrn-3136_56b515f95819974c3b586fd6b82a07b5', 'WebQTrn-2784_7084416ab9f72f1f1f8fc3ce7871ee4a', 'WebQTest-1840_20627dc086675c21450a79748f146c51', 'WebQTest-918_c544bd45cd8f0707031c1ae0dcf1b6f2', 'WebQTest-634_54d73a3ea0de9b701f865feec89dffd6', 'WebQTrn-25_5f5da3e1d4ca7df9f19ce3fdfc5790e2', 'WebQTrn-3543_3f03848605c6758ff2230a955cd92d65', 'WebQTest-1875_c932aa6e6365aa759f5d0f8da236643c', 'WebQTrn-1155_cb9805c4c3bedc59995289ea0f7dbf7f', 'WebQTest-1812_179b358fd4db2b454708a6448ec682de', 'WebQTrn-3151_f12c28a04e894937d193ae473c7dccab', 'WebQTrn-2540_663593327611e932c87fa55af9a84152', 'WebQTrn-3049_df62db039ae55ac085cf2731442925a1', 'WebQTest-1513_690504fdf4005195e5ad48d4570cf03f', 'WebQTrn-567_8e46718c3fc1361ff1c02b62a853b402', 'WebQTest-361_ea09afeb5a95dce3146077f968ff11f9', 'WebQTrn-1938_9440ac3bd6d8053ca4d553da408a7cb0', 'WebQTrn-567_291296777d1d44fe5794302704020cc8', 'WebQTrn-25_2a2de50d3b65cc5d2c88f54e283a4b8a', 'WebQTrn-2904_7fbcc8ac696c9cb8c1b0d451cd51ce4f', 'WebQTrn-25_3a7068fa96cd9e08dc2b1dabe458c85d', 'WebQTrn-1812_6a0e75033345508e8eadd5a47b0ad6b5', 'WebQTrn-2428_7f8dba203353c3b7569b073ccb09dae6', 'WebQTest-213_a2c4770244997a8496e0b8489aa442a9', 'WebQTrn-2215_4b1d9f87e9407bef5a468e6c1124c547', 'WebQTrn-60_7abf3620549d6d7aaf194ce37a34bbd8', 'WebQTest-1941_6ed17cf93677d92f2fcd2df541fa96b3', 'WebQTrn-783_3c516a15eba0156686ffd81866b60280', 'WebQTrn-1597_681d3cc1ea84472510a4fceba6d16eb4', 'WebQTrn-2047_3d5f855275265f74d81b862231578e33', 'WebQTrn-2904_2a7ddf9902ba184de2dace5671496b91', 'WebQTrn-303_cba5bd9bc905f6ca2f19b94bfc010b3a']
        
    poor_cwq_p4 = ['WebQTest-538_2055b38029bf8f2a0c593d3a189d3efe', 'WebQTrn-25_22837422e25cf0a05cb0efba9aa80402', 'WebQTrn-3249_20507e60ba33137383d2b85cb865826a', 'WebQTrn-465_e9585b7d3117fbe09c4ed03353acee7b', 'WebQTrn-62_632b1d8fd7ba66275f6f8d40ed044a13', 'WebQTrn-2721_2d207ed91313bd46c4ffd81c9e26a912', 'WebQTrn-303_770773a5150d1cdd3cfadfc25022720b', 'WebQTrn-634_8b1ff0551fb22f1d4e54d33d3656b8e8', 'WebQTrn-1770_540abec8ff3d2f4e81bfc5be9ea8e816', 'WebQTest-1923_d2a47301538ff61eb47d97dcd2b65863', 'WebQTrn-1532_118a81ac3fc08ca8590bf8be836d1be1', 'WebQTrn-2653_b98f847032635d8795209af243ad72b1', 'WebQTrn-2904_56cf05bd5b41e04f4358490ed3e81619', 'WebQTrn-60_bf5ce242e170352050a5b7308531e009', 'WebQTrn-60_039556a866f620ae9a1a72f96c22bae1', 'WebQTrn-2026_96417e4d721f5c0e2c9079454df0af5b', 'WebQTest-55_54e856d3c44e0ac4a33f0f0ebb7a67d6', 'WebQTrn-60_904dc2251690e2d7ad9328706488eb2f', 'WebQTrn-3166_cf9ffbb57f2b9e73dca2d2492121f242', 'WebQTrn-1758_213cbfafc2612b42c5f8efe85c3532c5', 'WebQTest-832_a188aff4a054e1ec66fafba1b8021f67', 'WebQTest-931_95ba50d5475eff7d8ffb3b3a36abbf97', 'WebQTrn-60_6bba574f25812fe575c83a65f6ec0ab8', 'WebQTrn-3049_c2209731c36910d996455a986525aba1', 'WebQTrn-2189_3b676cf5603d91ac09fd09d3697b68fb', 'WebQTest-1785_bd2684abe96767b0564cea78024983f2', 'WebQTrn-3084_be9304e56c1159daf6cad0a54de15eec', 'WebQTrn-724_d133cd1308f3fd8e12b6e2eb7acbc859', 'WebQTest-1923_7084416ab9f72f1f1f8fc3ce7871ee4a', 'WebQTest-1785_e7d9ebfefbe94a1c97fcf7df033acbe9', 'WebQTrn-2316_309331514903d37848eff7694e047856', 'WebQTrn-125_9ad09040a8f87a2abeca21f7311fd71e', 'WebQTrn-1677_5c50821cd29a183f2322fa2dd597a86c', 'WebQTrn-60_ab488396f4fb0104563e58b22214040c', 'WebQTrn-1278_d127c340850ac5ecd6e6f7892ca8f509', 'WebQTest-1923_bfd1c0770085dad4d2c23a21593c3738', 'WebQTest-1785_d74d32a8c6a9512c22aff68c266ee9c4', 'WebQTrn-3543_b5eabb006685da5755fdf968500b7cd4', 'WebQTest-1817_7c244b7098cc80194cf19c305455572e', 'WebQTest-1923_eaeaff16b51e52669edc90501a373c61', 'WebQTrn-25_4683684129d982885e4ac023f88bdad3', 'WebQTest-1379_8d8a89d2b3c70cf221c515e27f633b5b', 'WebQTest-450_b68e8f1a3f19f80d7dfc10bf3796b71e', 'WebQTest-699_c03c4525d4ac4e6fecf42cfd6fe1929a', 'WebQTrn-2653_428127267b489c72416c9298ec80bd7d', 'WebQTrn-2006_7993a79baca778d4e8e6cb2b1882bca4', 'WebQTrn-2429_4449720a6e25f3093a8065ae980f0221', 'WebQTrn-738_4c8856ea6b11794a061053bd4ee35371', 'WebQTrn-2818_7337f6e256ea9b695d810136db74f041', 'WebQTest-361_696a68cf3cd49341ba9a2ad99b6a9e59', 'WebQTrn-567_97c01fdfa60e1bc796af5839451cd9af', 'WebQTest-361_9a6514e92141770cd4fc2592c7768e4b', 'WebQTrn-464_cab1228be4e8c16ee35a67b7ac63b264', 'WebQTrn-1232_ae9a833cf2cca30c1c9725fc6a167794', 'WebQTrn-95_2ef516a3253d8d1d2c28ae11c6d5b2e9', 'WebQTrn-60_66d9f78b8294956373ed9ecbab3318a3', 'WebQTrn-2748_7712cb150914e7be807253ecc9fa6e3e', 'WebQTrn-493_85f381b8012b4def553954906bc9fafb', 'WebQTest-1923_29c81279ed9a982e12f82e764083db76', 'WebQTrn-2069_5a7f5c1dfeaddda67fda5178c6c27e16', 'WebQTrn-2250_98aa1c3bf990bf3d779dee7b611c33fa', 'WebQTrn-1283_e831da3802943dad506eb1e3fb611847', 'WebQTrn-2209_7bc37f5ea0bc419b7ee5510daff240df', 'WebQTest-55_aa809c4b556066af564266fe1f37ee4c', 'WebQTrn-1812_04dbed66ba17d35bf79f72e04bbcc776', 'WebQTrn-750_523a8d258a32c689c68cad6d2650dbc0', 'WebQTest-537_8ed4ed5b25424f7584947d0926dd6119', 'WebQTrn-3251_e7bfd0826590a74bd0a33bd2732849b7', 'WebQTrn-2286_6f8a4ec144197d617a362ba8798b18d0', 'WebQTest-1528_5d11d9c9cbbc8558d1d4af742af87b04', 'WebQTrn-261_29ca7af654dd447d9eccbc99846d80b2', 'WebQTrn-846_acc6d2d37e15db911023be3a597053e9', 'WebQTrn-2871_8894c3e2a3bfe181370fe33175dde864', 'WebQTest-212_7905d5d52ac1a17f68996d4a2245e682', 'WebQTrn-1283_b27c17c9e095dc8b429a20aeac481682', 'WebQTrn-2286_a23d2f977b0719cd9eb684929c371ccb', 'WebQTrn-3166_5a38185606a7a8c9158d615dddb5b29f', 'WebQTest-1376_6ccb0247e2b78057c0bd25b1539df404', 'WebQTrn-124_8a391bb9366c22ce0aadc00cfecf7e08', 'WebQTrn-567_774800570524cfa60e44ba5d6c56a02c', 'WebQTrn-1706_16e47af89161d355ca2d481aa51761a4', 'WebQTest-1840_c00f449746ccac175d49bbcd8f9c7757', 'WebQTest-1513_a60bea07c81573c12c93e515b219c8b0', 'WebQTrn-3136_8637ef3f86746bed90cde00b92471008', 'WebQTest-654_b183c990916cd703ef71720d378b748e', 'WebQTrn-1023_efcc9747755b72a5f927c5ccf984d9e4', 'WebQTrn-42_066342f54b4b8508e0b726af7fd7d91c', 'WebQTest-55_5d347c9ca6d42c2ca74e06b6bec081cb', 'WebQTrn-706_6032efd4798dd7c25582a0bc6cc5b4ea', 'WebQTrn-1770_79714129f735fe30ef12c7dc45814f91', 'WebQTrn-3136_b1b88f9e0dba1ce22772ea94c3596893', 'WebQTest-213_a301052c4da71e75def5ad8b69f9b06c', 'WebQTrn-2069_cf3638075fd9204c82c8adf9ae47925e', 'WebQTest-1306_1514f1e9c4973110faa2ad25559c0435', 'WebQTrn-493_487e4f4ea07d526a8acd19ffa9c542a8', 'WebQTest-1686_b30993b3354e5b4bb3ded1e07150aef8', 'WebQTrn-2069_755bf60e299967df7b265e22e1ac2367', 'WebQTrn-2664_d447122eab9b3a1bb5e1eeb45edaa3ef', 'WebQTrn-783_92242d9331d9137db2130a09859255de', 'WebQTrn-2311_c00aa883a19ff0f98199e4bc12c94dca', 'WebQTrn-567_d903bf47d5683f31ebe9f50099ecba09', 'WebQTrn-25_2055b38029bf8f2a0c593d3a189d3efe', 'WebQTest-1379_fabe39ef92b4822ff61ce7719accda87', 'WebQTrn-2047_bf0ea964c61fa149135eccfcefa002e6', 'WebQTrn-3151_5ff5b8826574b331ba342518d1c1bb34', 'WebQTrn-1069_b526307856da0d2f8d1bf4ceecb179ad', 'WebQTrn-3249_ba23107deaeaf5733abcc2cabc5221b6', 'WebQTrn-2653_aa9cd213dd7b3a0d3e98fdf230b024e3', 'WebQTest-55_033341263fdfc6fffe931632a3a0d95c', 'WebQTrn-1864_ed598699c0c39f5b0edfd4d64bb14e92', 'WebQTest-576_495085ad9dd6274cce483df63474ca21', 'WebQTrn-124_f9991dde9526307378d407546d8d8b1f', 'WebQTrn-241_fa1fffe7995213b6528da57ea4c8d226', 'WebQTrn-105_6ca8cba7511811830a04cd64a8a4cf77', 'WebQTrn-2754_a52b59230c3401a9e65db7fbf5acf8fa', 'WebQTrn-567_af0cf33a6bff9c43bc329d80c5ac4776', 'WebQTest-213_f063138e1c76517340f69d9b9181ea63', 'WebQTest-1797_f2a2e2e91494abbc29bdf70800a5910e', 'WebQTrn-3249_75f7d39d822bda23a91efa98ddabf6e8', 'WebQTrn-2779_9e9e6d9329b2d9d85b0faf6a496151a8', 'WebQTest-1802_0b1efb954250d417c58d39ccd891b696', 'WebQTrn-1283_fe799f0968328608acc72a24be05ffe2', 'WebQTrn-60_b0f2e52b3de9c267409fcfc2114271b1', 'WebQTrn-2748_4d6c65004ee28d487bc3921f625106b5', 'WebQTrn-2428_1a389cac92fb403d4a2f31b2833825fa', 'WebQTrn-1677_4c971bef32fb5777fb8b18ead5a702fe', 'WebQTrn-2540_9ed4dcee2e3a2bd52f3a33594726bb51', 'WebQTrn-2292_8784ca0ffa5adf55625c8af4f6b657b3', 'WebQTrn-1677_ce04a76110500f26ec58e625cb97c0ed', 'WebQTrn-2069_52c8e056ec6add6f53e03ba70667b590', 'WebQTrn-25_401347365692c5a69ad4c61e03ef4caf', 'WebQTrn-2189_b183c990916cd703ef71720d378b748e', 'WebQTrn-567_724a3b769671b1ea52a76af3a90687f4', 'WebQTrn-3249_3a7068fa96cd9e08dc2b1dabe458c85d', 'WebQTrn-25_b24903d6b3827266b8138a7036dcbb67', 'WebQTrn-60_0d1a0cade1b36ef4b0e60edd9af0bda2', 'WebQTrn-3033_c6146585b4ccc06a7cd611c93cba660c', 'WebQTest-55_350cad074495ce796cd95a9bbce0ceb4', 'WebQTrn-662_593632bf477d73bb33a311728d6e6f29', 'WebQTrn-2784_699eef038aecb4af8f113ca8cd4081a0', 'WebQTrn-3249_5ae0074bd07e87487226dfdcf95a4c5c', 'WebQTrn-1938_0e945cac8043fe5af615e4b2f0ddac8f', 'WebQTrn-3543_bbb0c8aa3a2941db5bf85e7557241fda']
    
    io_system = IO_System(args=arguments, tokenizer=tokenizer, model=model)
    # pdb.set_trace()
    for i in tqdm(range(0, 120)):
        start_time = get_system_time()
        print(f'<start tree search at {start_time}>\n')
        try:
            # solve graph type
            if not arguments.use_freebase:
                print(f'Begin to solve the problem {i+1}...\n')
                data = dataset[i]
                question = data['question']
                topic_entity_list = data['q_entity']
                answer_entity_list = data['a_entity']
                qid = data['id']
                ################################################# bad case #####################################################
                if qid in poor_case:
                    continue
                else:
                ################################################# normal case ###################################################
                    output_tree_json = defaultdict(lambda: defaultdict(list))
                    output_tree_json = {'qid': qid, 'topic_entity_list':topic_entity_list, 'question': question, 'answer': data['a_entity'], 'is_legal':True}
                    if not is_legal_data(topic_entity_list=topic_entity_list, answer_entity_list=answer_entity_list, graph=data['graph']): # 这里省去错误的数据 图中找不到的
                        print('*' * 40, f'问题: {question}是非法的, qid: {qid}','*' * 40)
                        output_tree_json['is_legal'] = False
                        tree_list.append(output_tree_json)
                        continue
            
            # solve freebase type 
            if arguments.use_freebase:
            ##################################################freebase读取方式##############################################################
                print(f'Begin to solve the problem {i+1}...\n')
                data = dataset[i]
                # data format => { question: string (with '?'), qid: str, id2topic_entity_list: list: [{mid:ent_str}, ...], answer_entity_list: [{mid:ent_str}, ....] webqsp graliqa, [ent_str, ent_str, ....]cwq }
                question, qid, id2topic_entity_list, answer_entity_list, answer_mid_list = parse_freebase_data(data=data, 
                                                                                                               task_name=arguments.task_name,
                                                                                                               question_string=question_string,
                                                                                                               q_string=q_string,)
                data_dict = {'qid':qid, 'question': question, 'id2topic_entity_list':id2topic_entity_list, 'answer_entity_list':answer_entity_list, 'answer_mid_list': answer_mid_list}
                
                # if qid not in poor_cwq_p4:
                #     continue
                # else:
                # ###### normal case #######
                if qid in poor_case:
                    continue
                else:
                # ###### normal case ######
                    output_tree_json = defaultdict(lambda: defaultdict(list))
                    output_tree_json = {'qid': qid, 'question': question, 'id2topic_entity_list': id2topic_entity_list, 'answer': answer_entity_list, 'answer_mid_list': answer_mid_list,}
            ##################################################freebase读取方式##############################################################    
            for id2topic_entity in id2topic_entity_list:
                # pdb.set_trace()
                mid, topic_entity = list(id2topic_entity.items())[0]
                print(f'Begin to solve the problem: {question}, \nmid: {mid}, \ntopic entity: {topic_entity}...\n')
                output_tree_json[topic_entity] = {}
                
                if arguments.mode == 'mcts':
                    Task = MCTS_Task(data=data_dict,
                                        topic_entity=topic_entity,
                                        mid=mid,
                                    # emb_model=emb_model,
                                    io_system=io_system, 
                                    propose_method=arguments.propose_method, 
                                    value_method=arguments.value_method,
                                    use_generator=arguments.use_generator, 
                                    end_gate=arguments.end_gate,
                                    roll_policy=arguments.roll_policy, 
                                    roll_branch=arguments.roll_branch, 
                                    num_plan_branch=arguments.num_plan_branch,
                                    num_branch=arguments.num_branch,
                                    sample_value=arguments.sample_value, 
                                    roll_forward_steps=arguments.roll_forward_steps, 
                                    time_limit=arguments.time_limit,
                                    iteration_limit=arguments.iteration_limit, 
                                    exploration_constant=arguments.exploration_constant, 
                                    alpha=arguments.alpha, 
                                    inf=arguments.inf,
                                    temperature=arguments.temperature, 
                                    use_reflection=arguments.use_reflection, 
                                    max_tokens=arguments.max_tokens, 
                                    max_new_tokens=arguments.max_new_tokens, 
                                    max_length=arguments.max_len, 
                                    try_num=arguments.try_num, 
                                    min_iteration_limit=arguments.min_iteration_limit, 
                                    max_child_num=arguments.max_child_num,
                                    low=arguments.low, 
                                    high=arguments.high, 
                                    limited_depth=arguments.limited_depth,
                                    use_vllm=arguments.use_vllm,
                                    shuffle=arguments.shuffle,
                                    shuffle_times=arguments.shuffle_times,
                                    use_rank_prompt=arguments.use_rank_prompt)
                    # pdb.set_trace()
                    is_in_graph, finish, root, path_with_reward, subquestions_list = Task.run()
                    if is_in_graph == False:
                        question = data['question']
                        print('*' * 40, f'本条数据的主题实体{topic_entity}不在子图中, 该问题是{question}')
                        continue
                root.trace_path()
                root.count_node()
                treeNode.reset_class_variable()
                
                output_tree_json[topic_entity]['steps'] = root.tree_list
                output_tree_json[topic_entity]['node_num'] = root.node_num
                output_tree_json[topic_entity]['maxdepth'] = root.maxdepth
                output_tree_json[topic_entity]['subquestions'] = subquestions_list
            tree_list.append(output_tree_json)
                # if arguments.visualize:
                #     visualize(root, Task, arguments.task_name, arguments.file, i + 1)

            print(f'The tree to problem {i+1} is complete.\n')
            end_time = get_system_time()
            print(f'<end tree search at {end_time}>\n')
            base_dir = os.getcwd()
            output_dir = pathlib.Path(f'{base_dir}/outputs/{arguments.task_name}/{Task.mode}/{Task.propose_method}')
            output_file = f'{base_dir}/outputs/{arguments.task_name}/{Task.mode}/{Task.propose_method}/{Task.propose_method}-{arguments.shuffle_times}-{arguments.num_plan_branch}-{arguments.num_branch}_{current_month}_{current_day}_{current_hour}_{current_minute}_{current_second}_alltree.json'
            pathlib.Path.mkdir(output_dir, exist_ok=True, parents=True)
            dump_json(output_file, tree_list)
        ################################################# normal case ###################################################
        except Exception as e:
            print(f"本条数据生成失败 下一条生成!\nError type:{e}\n")
            print(traceback.format_exc())
            end_time = get_system_time()
            print(f'<end tree search at {end_time}>\n')
            continue

    print('_' * 60)
    # accuracy

    if args.evaluate:
        print(f'Test accuracy:{correct_count / data_len}\n')
        print(f'Correct number of problems:{correct_count}\nTotal number of questions:{data_len}\n')
    print('_' * 60)



def parse_args():
    base_args = argparse.ArgumentParser()
    base_args.add_argument('--task_name', type=str, default='cwq')
    base_args.add_argument('--emb_model', type=str, choices=['gte', 'sentence_bert', 'text2vec'], default='gte')
    base_args.add_argument('--use_local_method', type=bool, default=True)
    base_args.add_argument('--propose_method', type=str, choices=['gpt', 'llama3', 'llama3.1', 'qwen7b', 'qwenapi', '4o-mini', 'qwen14b', 'qwenqwq', 'qwen32b'], default='qwen14b')
    base_args.add_argument('--value_method', type=str, choices=['gpt', 'llama', 'qwen', 'qwen', '4o-mini'], default='qwen')
    base_args.add_argument('--do_sample', type=bool, default=False)
    base_args.add_argument('--num_plan_branch', type=int, default=6)
    base_args.add_argument('--num_branch', type=int, default=3)
    base_args.add_argument('--truncation', type=bool, default=True)
    base_args.add_argument('--use_generator', type=bool, default=False)
    base_args.add_argument('--limited_depth', type=int, default=5)
    base_args.add_argument('--sample_value', type=str, choices=['simple', 'full'], default='full')
    base_args.add_argument('--mode', type=str, choices=['cot', 'tot', 'mcts'], default='mcts')
    base_args.add_argument('--temperature', type=float, default=0.7)
    base_args.add_argument('--time_limit', type=int, default=None)
    base_args.add_argument('--iteration_limit', type=int, default=2)
    base_args.add_argument('--roll_policy', type=str, choices=['random', 'greedy'], default='greedy')
    base_args.add_argument('--exploration_constant', type=float, default=0.4)
    base_args.add_argument('--roll_forward_steps', type=int, default=2)
    base_args.add_argument('--end_gate', type=float, default=0.95)  # End threshold
    base_args.add_argument('--roll_branch', type=int, default=1)
    base_args.add_argument('--max_len', type=int, default=30000)
    base_args.add_argument('--max_tokens', type=int, default=16000)
    base_args.add_argument('--try_num', type=int, default=5)
    base_args.add_argument('--max_new_tokens', type=int, default=256)
    base_args.add_argument('--inf', type=float, default=0.8)
    base_args.add_argument('--evaluate', type=bool, default=False)  # Whether to evaluate (empty means no evaluation)
    base_args.add_argument('--alpha', type=float, default=0.5)
    base_args.add_argument('--visualize', type=bool, default=False)  # visualization
    base_args.add_argument('--use_reflection', type=str, choices=['simple', 'common'], default='simple')  # Use reflective mode
    base_args.add_argument('--low', type=float, default=0)
    base_args.add_argument('--high', type=float, default=1)
    base_args.add_argument('--select_branch', type=int, default=2)
    base_args.add_argument('--max_depth', type=int, default=8)
    base_args.add_argument('--select_method', type=str, choices=['greedy', 'sample'], default='greedy')
    base_args.add_argument('--min_iteration_limit', type=int, default=86)
    base_args.add_argument('--max_child_num', type=int, default=9)
    base_args.add_argument('--use_vllm', default=False, action="store_true")
    base_args.add_argument('--shuffle', default=False, action='store_true')
    base_args.add_argument('--shuffle_times', type=int, default=1)
    base_args.add_argument('--use_rank_prompt', default=False, action='store_true')
    base_args.add_argument('--use_freebase', default=True, action='store_true')
    # base_args.add_argument('--model_id', type=int, default=4)
    arguments = base_args.parse_args()
    return arguments


if __name__ == '__main__':
    args = parse_args()
    run(args)
    # set_model(args.model_id)
