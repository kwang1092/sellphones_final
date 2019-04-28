import numpy as np
import csv
import re
import json
from textblob import TextBlob
from collections import defaultdict
from nltk.tokenize import TreebankWordTokenizer
import Levenshtein

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
        return(float(lst[2]))
    else:
        return None

def convertGB(num, byte):
    if byte.lower()=="mb":
        return float(num)/1024
    else:
        return num

def main():
    phones = {}
    labels = []
    with open('gsmphones.csv', mode='r') as csv_file:
        csv_reader = csv.reader(csv_file)
        for i,row in enumerate(csv_reader):
            if i == 0:
                labels = row
            else:
                phones[row[1]] = row[2:]

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
        else:
            phones[phone][label_to_index["price"]] = 100.0

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

    rel_labels = [label_to_index["model image"],label_to_index["announced"],label_to_index["thickness"],
              label_to_index["sim"],label_to_index["size"],
              label_to_index["cpu"],label_to_index["ppi"],label_to_index["card slot"],
              label_to_index["storage"],label_to_index["rear camera"],label_to_index["video"],
              label_to_index["3.5mm jack"],label_to_index["price"],label_to_index["dual"],
              label_to_index["front camera"],label_to_index["ram"]]

    features = {}
    for i,phone in enumerate(phones):
        features[phone] = []
        for label in rel_labels:
            features[phone].append(phones[phone][label])
    feature_mat = np.zeros((len(phones),len(features["Apple iPhone XS Max"])-1))
    feat_to_index = {"announced":0,"thickness":1,"sim":2,"size":3,"cpu":4,
                 "ppi":5,"card slot":6,"storage":7,"rear camera":8,
                 "video":9,"3.5mm jack":10,"price":11,"dual":12,
                 "front camera":13,"ram":14}
    phone_to_index = {}
    index_to_phone = []
    for i,phone in enumerate(features):
        feature_mat[i] = features[phone][1:]
        phone_to_index[phone] = i
        index_to_phone.append(phone)
    prices = feature_mat[:,feat_to_index["price"]]
    zero_200 = np.where(prices<200)[0]
    two_four = np.intersect1d(np.where(prices>=200)[0],np.where(prices<400)[0])
    four_six = np.intersect1d(np.where(prices>=400)[0],np.where(prices<600)[0])
    six_eight= np.intersect1d(np.where(prices>=600)[0],np.where(prices<800)[0])
    eight_ten= np.intersect1d(np.where(prices>=800)[0],np.where(prices<1000)[0])
    ten_12   = np.intersect1d(np.where(prices>=1000)[0],np.where(prices<1200)[0])
    luxury   = np.where(prices>=1200)[0]

    phonenames = list(phones.keys())

    brand_dic = {phone.split(" ")[0]:[] for phone in phones}
    for i,phone in enumerate(phones):
        phone_name = ""
        for word in phone.split(" "):
            phone_name += word+ " "
        brand_dic[phone.split(" ")[0]].append(phone_name[:len(phone_name)-1])

    brands = list(brand_dic.keys())

    json_file = open('reviews.json')
    json_str = json_file.read()
    reviews = json.loads(json_str)

    def edit_distance(query_str, name):
        return Levenshtein.distance(query_str.lower(), name.lower())

    def edit_distance_search(query, names):
        result = []
        for name in names:
            score = edit_distance(query,name)
            result.append((score,name))
        result = sorted(result, key=lambda tupl: tupl[0])
        return result

    userreviews = {} # dictionary where keys are phone names and values are list of reviews
    for p in list(reviews.keys()):
        words = p.split(" ")
        brand = words[0]
        if brand in brands:
            query = ""
            for i in range(0,5):
                if words[i] != "-":
                    query += p.split(" ")[i] + " "
            query = query[:len(query)-1]
            topmatch = edit_distance_search(query, brand_dic[brand])[0][1]
            if topmatch in list(userreviews.keys()):
                userreviews[topmatch][1] += reviews[p][1]
            else:
                userreviews[topmatch] = [reviews[p][0], reviews[p][1]]
    for p in phonenames:
        reviewed = list(userreviews.keys())
        if p not in reviewed:
            userreviews[p] = [0.0, []]

    reviewscores = {} # dictionary where keys are phone names and values are avg scores of reviews
    for p in phonenames:
        rating = float(userreviews[p][0])
        reviewlist = userreviews[p][1]
        n_reviews = len(reviewlist)
        if n_reviews != 0:
            total_polarity = 0.0
            rating_effect = 1.0

            if rating <= 1.0:
                total_polarity = -1.0
            elif rating <= 2.0:
                total_polarity = -0.7
            else:
                for r in reviewlist:
                    review = TextBlob(r)
                    total_polarity += review.sentiment.polarity

            ratio = rating/5.0
            if rating >= 4 and total_polarity > 0:
                rating_effect = 1.5*ratio
            elif rating >= 4 and total_polarity < 0:
                rating_effect = -1.5*ratio
            elif rating > 2 and total_polarity > 0:
                rating_effect = -1.0*ratio
            elif rating > 2 and total_polarity < 0:
                rating_effect = 1.5+(1-ratio)

            reviewscores[p] = rating_effect*total_polarity/n_reviews
        else:
            reviewscores[p] = 0.0

    ##query_feat = ["ram","front camera","cpu","rear camera"]
    ##price_range= ten_12

    query_vec = np.zeros(len(query_feat))
    for i,feat in enumerate(query_feat):
        query_vec[i] = max(feature_mat[:,feat_to_index[feat]])

    query_mat = feature_mat[price_range][:,[feat_to_index[feat] for feat in query_feat]]
    prange_to_phone = [index_to_phone[i] for i in price_range]

    cossim_vec = np.zeros(len(phones))
    for i,vec in enumerate(query_mat):
        vec /= query_vec
        cossim_vec[i] = np.linalg.norm(vec)
        brand = prange_to_phone[i].split(" ")[0]
        if brand == "Apple" or brand == "Samsung":
            cossim_vec[i] *= 1.5
        cossim_vec[i] *= 1+reviewscores[prange_to_phone[i]]

    rankings = np.where(cossim_vec>0)
    rankings = np.argsort(cossim_vec[rankings])[::-1]
    for i in range(1,min(21,len(price_range))):
        print(str(i)+". ", prange_to_phone[rankings[i-1]], cossim_vec[rankings[i-1]])

main()
