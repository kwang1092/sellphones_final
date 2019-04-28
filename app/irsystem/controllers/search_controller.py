from . import *
from app.irsystem.models.helpers import *
from app.irsystem.models.helpers import NumpyEncoder as NumpyEncoder
import numpy as np
import csv
import re
import json
from textblob import TextBlob
from collections import defaultdict
from nltk.tokenize import TreebankWordTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse.linalg import svds
import Levenshtein
from sklearn.preprocessing import normalize

apple_mult = 1.24
project_name = "sellPhones"
net_id = "Alvin Qu: aq38, Andrew Xu: ax28, Kevin Wang: kw534, Samuel Han: sh779"

@irsystem.route('/', methods=['GET'])
def search():
    check = False
    mate = False
    past = request.args.get('past')
    past2 = request.args.get('past2')
    if past:
        check = True
        mate=True
    else:
        mate = True

    if past2 == "True":
        flag = True
    else:
        flag = False

    input_arr = []
    final = [[0],[0]]

    condition = request.args.get('condition')
    budget = request.args.get('budgets')
    feature_list = request.args.getlist('feature')
    old_phone = ""
    old_phone = request.args.get('old_phone')
    feature_text = request.args.get('feature_text')

    if not feature_list:
        return render_template('search.html', name=project_name,netid=net_id, check=check,  mate=mate, flag=flag,
                                condition=condition, names=[], urls = [], budget=str(budget))

    if budget and feature_list and condition:


        def lenCheck(dic):
            init = 0
            for i,elt in enumerate(dic):
                if i == 0:
                    init = len(dic[elt])
                if len(dic[elt])!=init:
                    return False
            return True

        def getResolution(res):
            lst = []
            if len(res)>=12:
                lst = re.findall(r"\d+\.?\d?",res)
            if len(lst)>1:
                return (int(lst[0]),int(lst[1]))
            else:
                return (0,0)

        def getDiagonalSize(size):
            if len(size)>=10:
                return float(re.findall(r"\d+\.?\d?",size)[0])

        def getThickness(dim):
            lst = re.findall(r"\d+\.?\d+",dim)
            if len(lst)>2:
                return(100-float(lst[2]))
            else:
                return None

        def convertGB(num, byte):
            if byte.lower()=="mb":
                return float(num)/1024
            else:
                return num

        def edit_distance_search(query,names):
            result = []
            for name in names:
                score = Levenshtein.distance(query.lower(), name.lower())
                result.append((score,name))
            result = sorted(result, key=lambda x: x[0])
            return result

        def main(budget, feature_list,condition):
            phones = {}
            labels = []
            with open('app/static/gsmphones.csv', mode='r',encoding='utf-8') as csv_file:
                csv_reader = csv.reader(csv_file)
                for i,row in enumerate(csv_reader):
                    if i == 0:
                        labels = row
                    else:
                        phones[row[1]] = row[2:]

            phones.pop("Samsung Galaxy S10 5G")
            for elt in phones:
                l = len(phones[elt])
                if l<100:
                    while(len(phones[elt])<100):
                       phones[elt].append(0)


            label_to_index = {labels[i].lower(): i-2 for i in range(len(labels))}
            label_to_index["ppi"] = 81
            label_to_index["storage"] = 82
            label_to_index["ram"] = 83
            label_to_index["rear camera"] = 85
            label_to_index["front camera"] = 86
            new_phones = {}

            for i,phone in enumerate(phones):
                a,b = getResolution(phones[phone][label_to_index["resolution"]])
                size= getDiagonalSize(phones[phone][label_to_index["size"]])
                phones[phone][label_to_index["size"]] = size
                diag= np.sqrt(a**2+b**2)
                if size!=None and size>=1.7 and size<=7.0 and a!=0 and b!=0:
                    phones[phone][label_to_index["ppi"]] = diag/size
                    card_slot = phones[phone][label_to_index["card slot"]]
                    if card_slot.lower()=="no":
                        phones[phone][label_to_index["card slot"]] = 0
                    else:
                        phones[phone][label_to_index["card slot"]] = 1
                    new_phones[phone] = phones[phone]
            phones = new_phones

            new_phones = {}
            for i,phone in enumerate(phones):
                dim = phones[phone][label_to_index["dimensions"]]
                if len(dim) >= 12 and "mm" in dim:
                    phones[phone][label_to_index["dimensions"]] = getThickness(dim)
                    new_phones[phone] = phones[phone]
            phones = new_phones
            assert(lenCheck(phones))

            new_phones = {}
            for i,phone in enumerate(phones):
                lst = re.findall(r"[\d.]+",phones[phone][label_to_index["internal"]])
                byte= re.findall(r"[a-zA-Z]+",phones[phone][label_to_index["internal"]])
                if len(lst)>1:
                    mem = 0
                    ram = 0
                    if "ROM" in byte and "RAM" in byte:
                        ram = convertGB(lst[-2],byte[byte.index('RAM')-1])
                        mem = convertGB(lst[-1],byte[byte.index('ROM')-1])
                    else:
                        try:
                            ram = convertGB(lst[-1],byte[byte.index('RAM')-1])
                        except:
                            ram = convertGB(lst[-1],byte[-1])
                        if float(lst[-1]) >= 100:
                            lst.pop(len(lst)-1)
                        lst = [float(elt) for elt in lst]
                        mem = convertGB(sorted(lst)[-1],byte[0])
                    phones[phone][label_to_index["storage"]] = float(mem)
                    phones[phone][label_to_index["ram"]] = float(ram)
                    new_phones[phone] = phones[phone]
                elif len(lst)==1 and float(lst[0]) > 0:
                    phones[phone][label_to_index["storage"]] = float(convertGB(lst[-1],byte[0]))
                    phones[phone][label_to_index["ram"]] = 0
                new_phones[phone] = phones[phone]
            phones = new_phones
            assert(lenCheck(phones))
            new_phones = {}

            for i,phone in enumerate(phones):
                temp = phones[phone][label_to_index["price"]]
                price = re.findall(r"\d+\.?\d?",temp)
                curr = re.findall(r"[a-zA-Z]+",temp)
                if len(temp) > 0:
                    if curr[-1].lower()=="eur":
                        phones[phone][label_to_index["price"]] = round(float(price[0]) * 1.13, 2)
                    else:
                        phones[phone][label_to_index["price"]] = round(float(price[0]) * 0.014, 2)
                    new_phones[phone] = phones[phone]
            phones = new_phones

            new_phones = {}
            for i,phone in enumerate(phones):
                sim = phones[phone][label_to_index["sim"]].lower()
                if "dual" in sim:
                    phones[phone][label_to_index["sim"]] = 1
                else:
                    phones[phone][label_to_index["sim"]] = 0

                temp = phones[phone][label_to_index["video"]]
                fps  = re.findall(r"\d+[a-zA-z][a-zA-z]+",temp)
                if len(fps)==0:
                    fps = 0
                else:
                    fps = float(re.findall(r"\d+",fps[0])[0])
                temp = re.findall(r"\d+\.?\d?",temp)
                if len(temp) > 0:
                    phones[phone][label_to_index["video"]] = float(temp[0])*(fps/30)
                else:
                    phones[phone][label_to_index["video"]] = 240.0

                temp = phones[phone][label_to_index["announced"]]
                if type(temp) == int or type(temp) == float:
                    if temp >= 2018:
                        phones[phone][label_to_index["announced"]] = 1.0
                    else:
                        phones[phone][label_to_index["announced"]] = 0.0
                else:
                    if temp[:4] == "2019" or temp[:4] == "2018":
                        phones[phone][label_to_index["announced"]] = 1.0
                    else:
                        phones[phone][label_to_index["announced"]] = 0.0

                if phones[phone][label_to_index["3.5mm jack"]] == "No":
                    phones[phone][label_to_index["3.5mm jack"]] = 0.0
                else:
                    phones[phone][label_to_index["3.5mm jack"]] = 1.0

                temp = phones[phone][label_to_index["cpu"]].lower()
                if "octa" in temp:
                    phones[phone][label_to_index["cpu"]] = 8.0
                elif "hexa" in temp:
                    phones[phone][label_to_index["cpu"]] = 6.0
                elif "quad" in temp:
                    phones[phone][label_to_index["cpu"]] = 4.0
                elif "dual" in temp:
                    phones[phone][label_to_index["cpu"]] = 2.0
                else:
                    phones[phone][label_to_index["cpu"]] = 1.0

                dual = phones[phone][label_to_index["dual"]]
                front= re.findall(r"\d+",phones[phone][label_to_index["single"]])
                if len(front)>1 and ("mp" in phones[phone][label_to_index["single_1"]].lower() or
                                     "mp" in str(phones[phone][label_to_index["dual"]]).lower()):
                    cam1 = float(front[0])
                    if dual!="" and phone.split(" ")[0] != 'Acer':
                        rear = re.findall(r"\d+",phones[phone][label_to_index["dual"]])
                        phones[phone][label_to_index["dual"]] = 1
                        cam2 = float(rear[0])
                        phones[phone][label_to_index["front camera"]]= min(cam1,cam2)
                        phones[phone][label_to_index["rear camera"]] = max(cam1,cam2)
                    else:
                        rear = re.findall(r"\d+",phones[phone][label_to_index["single_1"]])
                        phones[phone][label_to_index["dual"]] = 0
                        cam2 = float(rear[0])
                        if len(rear)>0:
                            phones[phone][label_to_index["front camera"]]= min(cam1,cam2)
                            phones[phone][label_to_index["rear camera"]] = max(cam1,cam2)
                else:
                    phones[phone][label_to_index["front camera"]] = 0
                    phones[phone][label_to_index["dual"]] = 0
                    if len(front) > 0 and "mp" in phones[phone][label_to_index["single"]].lower():
                        phones[phone][label_to_index["rear camera"]] = float(front[0])
                    else:
                        phones[phone][label_to_index["rear camera"]] = 0

                new_phones[phone] = phones[phone]
            phones = new_phones

            label_to_index["thickness"] = label_to_index["dimensions"]
            label_to_index.pop("dimensions")

            assert(lenCheck(phones))

            rel_labels = [label_to_index["model image"],label_to_index["announced"],
              label_to_index["cpu"],label_to_index["ppi"],label_to_index["storage"],
              label_to_index["video"],label_to_index["dual"],label_to_index["ram"],
              label_to_index["thickness"],label_to_index["sim"],label_to_index["card slot"],
              label_to_index["rear camera"],label_to_index["front camera"],
              label_to_index["size"],label_to_index["3.5mm jack"],label_to_index["price"]]

            features = {}
            for i,phone in enumerate(phones):
                features[phone] = []
                for label in rel_labels:
                    features[phone].append(phones[phone][label])
            feature_mat = np.zeros((len(phones),len(features["Apple iPhone XS Max"])-1))

            feat_to_index = {"announced":0,"cpu":1,"ppi":2,"storage":3,
                 "video":4,"dual":5,"ram":6,"thickness":7,"sim":8,
                 "card slot":9,"rear camera":10,"front camera":11,
                 "size":12,"3.5mm jack":13,"price":14}

            phone_to_index = {}
            index_to_phone = []
            for i,phone in enumerate(features):
                feature_mat[i] = features[phone][1:]
                phone_to_index[phone] = i
                index_to_phone.append(phone)

            bin_feats = {"3.5mm jack":0,"dual":1,"sim":2,"announced":3,"card slot":4}

            #Need user input
            # min_price = 800
            # max_price = 1600
            # price_range = budget

            prices = feature_mat[:,feat_to_index["price"]]
            budget = str(budget)
            budget = budget.split(";")
            starting = int(budget[0])
            ending = int(budget[1])
            price_range = np.intersect1d(np.where(prices>=starting)[0],np.where(prices<ending)[0])

            # budget = int(budget)
            # if budget == 1:
            #     budget = zero_200
            # elif budget == 2:
            #     budget = two_four
            # elif budget == 3:
            #     budget = four_six
            # elif budget == 4:
            #     budget = six_eight
            # elif budget == 5:
            #     budget = eight_ten
            # elif budget == 6:
            #     budget = ten_12
            # else:
            #     budget = luxury

            reviewscores = {}
            with open('app/static/reviewscores.csv', mode='r') as csv_file:
                csv_reader = csv.reader(csv_file)
                for i,row in enumerate(csv_reader):
                    reviewscores[row[0]] = float(row[1])

            if condition == "new":
                condition = 1
            else:
                condition = 0
            query_feat = feature_list

            #query_feat = ["ram","front camera","cpu","rear camera"]
            # price_range= luxury

            if old_phone:
                old_query = old_phone
            else:
                old_query = ""

            # if not old_phone:
            #     old_phone = ""
            # old_query = old_phone
            results = {}
            ranked_results = []
            best_match = edit_distance_search(old_query,phones.keys())[0][1]
            best_dist = edit_distance_search(old_query,phones.keys())[0][0]
            best_match_vec = feature_mat[phone_to_index[best_match]]

            def checkBinary(idx):
                phone = index_to_phone[idx]
                for feat in query_feat:
                    if feat in bin_feats and feature_mat[idx][feat_to_index[feat]]!=1:
                        return False
                return True

            cossim = {}
            for p in price_range:
                if feature_mat[p][0] == condition and index_to_phone[p] != old_query and checkBinary(p):
                    temp = feature_mat[p,:]
                    cossim[p] = np.dot(temp,best_match_vec)/(np.linalg.norm(temp)*np.linalg.norm(best_match_vec))
            cossim_lst = []
            for idx in cossim:
                cossim_lst.append([idx,cossim[idx]])
            cossim_lst = np.array(cossim_lst)
            if len(cossim_lst)==0:
                return [[],[]]
            cossim_lst[:,1] /= max(cossim_lst[:,1])
            sim_index = np.argsort(cossim_lst[:,1])[::-1]
            cossim_min = min(cossim_lst[:,1])
            for idx in sim_index:
                results[index_to_phone[int(cossim_lst[idx][0])]] = (cossim_lst[idx,1]-cossim_min)/(max(cossim_lst[:,1])-cossim_min)

            query_vec = np.zeros(len(query_feat))
            for i,feat in enumerate(query_feat):
                query_vec[i] = max(feature_mat[:,feat_to_index[feat]])

            query_mat = []
            prange_to_phone = []
            for p in price_range:
                if feature_mat[p][0] == condition and index_to_phone[p] != old_query and checkBinary(p):
                    prange_to_phone.append(index_to_phone[p])
                    query_mat.append(feature_mat[p][[feat_to_index[feat] for feat in query_feat]])
            query_mat = np.array(query_mat)

            cossim_vec = np.zeros(len(phones))
            for i,vec in enumerate(query_mat):
                vec /= query_vec
                brand = prange_to_phone[i].split(" ")[0]
                if brand == "Apple":
                    vec *= apple_mult
                # print(prange_to_phone[i],vec)
                cossim_vec[i] = np.linalg.norm(vec)
                cossim_vec[i] *= 1+0.05*reviewscores[prange_to_phone[i]]

            rankings = np.where(cossim_vec>0)
            rankings = np.argsort(cossim_vec[rankings])[::-1]
            cossim_vec /= max(cossim_vec)


            if feature_text:
                custom_input_query = feature_text
                phonenames = list(phones.keys())

                brand_dic = {phone.split(" ")[0]:[] for phone in phones}
                for i,phone in enumerate(phones):
                    phone_name = ""
                    for word in phone.split(" "):
                        phone_name += word+ " "
                    brand_dic[phone.split(" ")[0]].append(phone_name[:len(phone_name)-1])

                brands = list(brand_dic.keys())
                brands.remove('T-Mobile')
                # brands.remove('AT&T')

                with open('app/static/userreviews.json', 'r') as fp:
                    userreviews = json.load(fp)

                with open('app/static/concat_reviews.json', 'r') as fp:
                    concat_reviews = json.load(fp)

                with open('app/static/tokenized_reviews.json', 'r') as fp:
                    tokenized_reviews = json.load(fp)

                lookaround_matrix = np.loadtxt('app/static/sent_anal_matrix.txt')

                review_vocab = []
                for phone,words in tokenized_reviews.items():
                    review_vocab += words

                #remove dups
                review_vocab = list(set(review_vocab))
                review_vocab.sort()

                review_phonenames = list(tokenized_reviews.keys())

                def build_inv_idx(lst):
                    """ Builds an inverted index.

                    Params: {lst: List}
                    Returns: Dict (an inverted index of phones)
                    """
                    inverted_idx = {}
                    for idx in range(0,len(lst)):
                        inverted_idx[lst[idx]] = idx
                    return inverted_idx

                review_vocab_invidx = build_inv_idx(review_vocab)
                review_names_invidx = build_inv_idx(review_phonenames)

                review_list = [concat_reviews[p] for p in concat_reviews]
                vectorizer = TfidfVectorizer(stop_words = 'english',encoding='utf-8',lowercase=True)
                my_matrix = vectorizer.fit_transform(review_list).transpose()
                u, s, v_trans = svds(my_matrix, k=60)
                words_compressed, _, docs_compressed = svds(my_matrix, k=40)
                docs_compressed = docs_compressed.transpose()
                word_to_index = vectorizer.vocabulary_
                index_to_word = {i:t for t,i in word_to_index.items()}
                words_compressed = normalize(words_compressed, axis = 1)
                def closest_words(word_in, k = 10):
                    if word_in not in word_to_index: return "Not in vocab."
                    sims = words_compressed.dot(words_compressed[word_to_index[word_in],:])
                    asort = np.argsort(-sims)[:k+1]
                    return [(index_to_word[i],sims[i]/sims[asort[0]]) for i in asort[1:]]

                def query_word(word):
                    close_words = closest_words(word)
                    if close_words == "Not in vocab.":
                        return word.split(" ")
                    return [word,close_words[0][0],close_words[1][0]]

                words_from_svd = []
                for word in custom_input_query.split(" "):
                    words_from_svd += query_word(word)
                n_words = len(words_from_svd)
                n_phones = len(review_phonenames)
                query_matrix = np.zeros((n_phones,n_words))

                new_string = []
                for word in words_from_svd:
                    if word in review_vocab:
                        new_string.append(word)

                #RANKINGS using custom input
                for abc in review_phonenames:
                    p = review_names_invidx[abc]
                    for i,word in enumerate(new_string):
                        w = review_vocab_invidx[word]
                        query_matrix[p,i] = lookaround_matrix[p,w]

                #Outputting ranking based on social component
                query_matrix = np.sum(query_matrix, axis=1)
                query_matrix = query_matrix / len(new_string)

                #WITH RATINGS (to be merged with cell above later)
                for phone in review_phonenames:
                    p = review_names_invidx[phone]
                    rating = float(userreviews[phone][0])
                    rating_effect = 1.0
                    ratio = rating/5.0
                    polarity = query_matrix[p]

                    if rating >= 4 and polarity > 0:
                        rating_effect = 1.3*ratio
                    elif rating >= 4 and polarity < 0:
                        rating_effect = -1.3*ratio
                    elif rating <= 2.5 and polarity > 0:
                        rating_effect = -1.0*ratio
                    elif rating <= 2.5 and polarity < 0:
                        rating_effect = 1.3+(1-ratio)
                    query_matrix[p] = rating_effect*polarity

                ranking_asc = list(np.argsort(query_matrix))
                ranking_desc = ranking_asc[::-1]

                test_dict = {}
                for i in ranking_desc:
                    test_dict[review_phonenames[i]] = query_matrix[i]


                if old_query != "":
                    for idx in rankings:
                        best_dist = edit_distance_search(old_query,[prange_to_phone[idx]])[0][0]
                        if prange_to_phone[idx] in test_dict:
                            if best_dist <= 5:
                                results[prange_to_phone[idx]] += cossim_vec[idx] + test_dict[prange_to_phone[idx]]/2
                            else:
                                results[prange_to_phone[idx]] += 3*(cossim_vec[idx] +  test_dict[prange_to_phone[idx]]/2)
                                results[prange_to_phone[idx]] /= 5
                        else:
                            if best_dist <= 5:
                                results[prange_to_phone[idx]] += cossim_vec[idx]
                            else:
                                results[prange_to_phone[idx]] += 3*(cossim_vec[idx])
                                results[prange_to_phone[idx]] /= 5
            else:
                print("yo")
                if old_query != "":
                    for idx in rankings:
                        best_dist = edit_distance_search(old_query,[prange_to_phone[idx]])[0][0]
                        if best_dist <= 5:
                            results[prange_to_phone[idx]] += cossim_vec[idx]
                        else:
                            results[prange_to_phone[idx]] += 3*(cossim_vec[idx])
                            results[prange_to_phone[idx]] /= 5

            final_rank = []
            for phone in results:
                final_rank.append((phone,results[phone]))
            final_rank = sorted(final_rank,key=lambda x: x[1])[::-1]
            result = []
            urls   = []
            for elt in final_rank:
                result.append(elt[0])
                urls.append(features[elt[0]][0])
            return [result,urls]

        final = main(budget,feature_list,condition)


    return render_template('search.html', name=project_name,netid=net_id, check=check,  mate=mate, flag=flag,
                            condition=condition, names=final[0], urls = final[1],budget=str(budget))
