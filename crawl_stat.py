import os
import numpy as np
import math
import nltk
from nltk.tokenize import RegexpTokenizer
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.mlab as mlab
from scipy.stats import norm

'''
    a quick analysis of some ToS crawl data and 
    snapshots of specifics along the different fields of 
    folks who require ToS.

    the crawl data contains several kinds of documents,
    I have also manually classified several of the companies [15]
    of each into lists which i will use as a reference for differences
    between genre. 

'''

#-------------- Document Groups -------------
'''
in the crawl data there are a bunch of synonymous
document names. I grouped them here

privacy: docs related to what a company does with your data

terms: terms of use, service, eula, etc.

legal jargon: basically when companies warn you how they will sue
you when the time comes.

third party: third party applications, advertisements, and data revenue.
'''

privacy = [
'Privacy Policy.txt','Amazon.com Privacy Notice.txt','Privacy Statement.txt',
'US Consumer Privacy Notice.txt','Customer Privacy Policy.txt','Web Services Privacy Policy.txt',
'Data Policy.txt','Privacy Policy 2.txt','Cookies Policy.txt','Privacy Policies.txt',
'Privacy Policy English (International).txt','Privacy Policy Agreement.txt'
]

terms = [
'Terms of Service.txt','Terms and Conditions.txt','Host Guarantee Terms and Conditions.txt',
'Amazon Kindle Store Terms of Use.txt','Conditions of Use.txt','Web Notices and Terms of Use.txt', 'Terms of Use and Privacy Policy.txt',
'iCloud Terms of Service.txt', 'iTunes Terms of Service.txt', 'Website Terms of Service.txt', 'Acceptable Use Policy.txt',
'Internet Terms of Service.txt', 'Wireless Plan Terms.txt','Visitor Agreement.txt', 'Terms of Use.txt',
'Copyright Notices and Counter-notices.txt', 'Copyright.txt', 'Disclaimer.txt', 'Vendor Terms of Service.txt',
'Copyright and Your use of the British Library Website.txt','Terms Of Service.txt', 'Policies.txt', 'Privacy.txt',
'Your Content Submissions.txt', 'Online Practices.txt','Google Project Hosting - Additional Terms.txt',
'Google Project Hosting - User Content and Conduct Policy.txt','Acceptable Use Policy for Xfinity Internet.txt',
'Residential Subscriber Agreement.txt', 'Software License Agreement.txt','Web Terms of Service.txt',
'Consumer Terms of Sale.txt','Etiquette Policy.txt', 'Policies and Notices.txt', 'Cable Internet Terms of Use.txt',
'Internet Service Agreement.txt', 'User Agreement.txt','Data Use Policy.txt', 'Additional Terms of Service.txt',
'Community Guidelines.txt', 'Universal Terms Of Service.txt', 'Google Analytics Terms of Service.txt',
'Terms of Service and License Agreement.txt', 'Residential Services Subscriber Agreement.txt','API Terms of Use.txt',
'Privacy Policy and Terms of Use.txt', 'Microsoft Services Agreement.txt','Terms of Use 2.txt','All Policies.txt',
'Terms of Service 1.txt', 'Terms of Service 2.txt', 'Terms of Service 3.txt', 'EULA.txt',
'Beleid voor Acceptabel Gebruik.txt', 'Global Acceptable Use Policy.txt', 'Nutzungsbedingungen.txt',
'Política de Uso Aceitável.txt', 'Política de uso aceptable.txt', 'US Acceptable Use Policy.txt',
'可接受使用政策.txt', 'Network User Agreement.txt', 'Business End User License Agreement (US).txt',
'Business End User License Agreement.txt','Etiquette.txt', 'Fair Usage Policy for Subscriptions.txt',
'Fair Usage Policy for US Minute Bundles.txt', 'Group Video Calling Fair Usage Policy.txt',
'Terms and Policies.txt', 'Terms of Service - Business (US).txt', 'Terms of Service - Business.txt',
'Terms and Conditions (Mobile).txt', 'Terms and Conditions (Premium).txt', 'Terms and Conditions (Unlimited).txt',
'Terms and Conditions of Use.txt', 'APIs Terms of Use.txt', 'Content Policy.txt','Steam Subscriber Agreement.txt',
'Legal Information.txt', 'Terms of Service and AUP.txt', 'Terms and Conditions and Privacy Policy.txt',
'Term of Service.txt', 'WordPress.com Terms Of Service.txt', 'World Of Warcraft Terms Of Use Agreement.txt',
'Blogger Terms Of Service.txt', 'Terms Of Use.txt', 'PCS Terms & Conditions.txt'
]

additional_legal_jargon = [
'Copyright Policy.txt','Amazon App Suite Legal Notices.txt','Kindle Cloud Reader Legal Notices.txt',
'Kindle for Mac Legal Notices.txt','Kindle for PC Legal Notices.txt', 'Kindle for Windows 8 Legal Notices.txt',
'Legal Disclaimer.txt','Security.txt','Fraud.txt', 'Civil Subpoena Policy.txt', 'Domain Name Proxy Agreement.txt',
'Domain Name Registration Agreement.txt', 'Trademark and Copyright Infringement Policy.txt', 
'Uniform Domain Name Dispute Resolution Policy.txt','Notice of Alleged Infringement.txt',
'Legal Information (Intuit).txt','Legal Notices.txt','Skype Emergency Calling.txt', 
'Trademark Guidelines.txt','Trademark Guidance.txt'
]
#additional copyright, subpoena, etc.

third_party = [
'Interest-Based Ads.txt','Third Party Licenses.txt',
'Third Party Advertising', 'Third Party', 'Cookies','Affiliation, and Terms of Usage.txt'
]

reference_dict = {'terms of service':terms,
                'privacy agreement':privacy,
                'additional legal terms':additional_legal_jargon,
                'third party agreements':third_party}

#-------------- Company Groups ---------------

'''
Todo:
    fill these lists with at least 30 files each
'''

social_media = ['facebook.com','twitter.com','instagram.com','pinterest.com',
'reddit.com','tumblr.com','myspace.com','okcupid.com']

retail = ['amazon.com','target.com','alibaba.com','adidas.com','basspro.com',
'barneys.com','crateandbarrel.com','safeway.com','walmart.com',
'gap.com','dickssportinggoods.com','ikea.com','officemax.com','orientaltrading.com']

geo_location = []

streaming = ['spotify.com','netflix.com','syfy.com','hulu.com','pandora.com']

search_engines = ['google.com','bing.com','yahoo.com','aol.com','duckduckgo.com','ask.com']

banks_and_finanical = ['wellsfargo.com','chase.com','venmo.com','citizensbank.com','squareup.com']

word_tokenizer = RegexpTokenizer(r'[\w\'.]+\w+')

sent_tokenizer = RegexpTokenizer(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s')

char_tokenizer = RegexpTokenizer(r'\w{1}')

PAGELENGTH = 350

def count_words(path):
    '''
    @param param1: path of the html file to be counted
    @return: an integer representation of the number of words in the document
    '''
    f = open(path,'r',encoding="utf-8")
    html = f.read()
    f.close()
    soup = BeautifulSoup(html,"html.parser")
    data = soup.find_all(text=True)
    data = "".join(data)
    words = len(word_tokenizer.tokenize(data))
    return words

def count_sentences(path):
    '''
    @param param1: file path
    @return: an integer representation of the number of sentences in the document
    '''
    f = open(path,'r',encoding="utf-8")
    html = f.read()
    f.close()
    soup = BeautifulSoup(html,"html.parser")
    data = soup.find_all(text=True)
    data = " ".join(data)
    words = len(sent_tokenizer.tokenize(data))
    return words

def count_characters(path):
    '''
    @param param1: file path
    @return: an integer representation of the number of chracters in the document
    '''
    f = open(path,'r',encoding="utf-8")
    html = f.read()
    f.close()
    soup = BeautifulSoup(html,"html.parser")
    data = soup.find_all(text=True)
    data = "".join(data)
    words = len(char_tokenizer.tokenize(data))
    return words

def get_doc_type(document_name):
    '''
    @param param1: the document name
    @return: a string indicating the classification of the document
    '''
    for key,value in reference_dict.items():
        if document_name in value:
            return key

def to_path(com_list):
    return list(['./web_crawl/'+f for f in com_list])

def map_from_list(list,dict):
    '''
    takes a list of values and adds them to a dict
    under the first key it is less than or equal to.

    for example: dict = {1:[],3:[],5:[]}
                 list = [1,1,3,4,5]
                 map_from_list(list,dict)
                 >>> dict
                 ... {1:[1,1],3:[3],5:[4,5]}

    @param param1: a list of integer values
    @param param2: a dict of lists with keys ranging from 0 to max(list)
    '''
    for list_val in list:
        dict[math.ceil(list_val)].append(list_val)
 

def overall_page_distribution():

    bar_graph_map = {}
    aggr = []

    #max page length found through experimentation
    for i in range(1,62):
        bar_graph_map[i] = []

    file_list = to_path(os.listdir("./web_crawl"))

    for fname in file_list:
        contents = [fname+'/'+text_file for text_file in os.listdir(fname)]
        page_count = [count_words(path)/PAGELENGTH for path in contents]
        page_count = list(filter(lambda x: x != 0.0 ,page_count))
        aggr += page_count
        map_from_list(page_count,bar_graph_map)
    
    mean = np.average(aggr)
    median = np.median(aggr)
    std = np.std(aggr)

    keys = bar_graph_map.keys()
    values = bar_graph_map.values()

    values = [len(val) for val in values]
    x = np.arange(len(keys))

    plt.bar(x, values,color="black")
    plt.ylabel('No. of Documents', fontsize=10)
    plt.xlabel('approximate length in pages', fontsize=10)

    mean_patch = mpatches.Patch(color='white', label='Mean: '+str(mean))
    std_patch = mpatches.Patch(color='white', label='Standard Dev: '+str(std))
    med_patch = mpatches.Patch(color='white', label='Median: '+str(median))

    plt.legend(handles=[mean_patch,std_patch,med_patch])
    plt.xticks(x, keys, fontsize=8, rotation=75)
    plt.show()


def calc_ari(path):
    words = count_words(path)
    sents = count_sentences(path)
    chars = count_characters(path)
    if 0 in [words,chars,sents]:
        return 0.0
    return 4.71*(chars/words)+0.5*(words/sents)-21.43

def ari_distribution():
    
    bar_graph_map = {}
    aggr = []

    #max page length found through experimentation
    for i in range(1,15):
        bar_graph_map[i] = []

    file_list = to_path(os.listdir("./web_crawl"))

    for fname in file_list:
        contents = [fname+'/'+text_file for text_file in os.listdir(fname)]
        page_count = [calc_ari(path) for path in contents]
        page_count = list(filter(lambda x: x > 0.0 and x <= 14 ,page_count))
        aggr += page_count
        map_from_list(page_count,bar_graph_map)
    
    mean = np.average(aggr)
    median = np.median(aggr)
    std = np.std(aggr)

    keys = bar_graph_map.keys()
    values = bar_graph_map.values()

    values = [len(val) for val in values]
    x = np.arange(len(keys))

    plt.bar(x, values,color="grey")
    plt.ylabel('No. of Documents', fontsize=10)
    plt.xlabel('Automated Readbility Index', fontsize=10)

    mean_patch = mpatches.Patch(color='white', label='Mean: '+str(mean))
    std_patch = mpatches.Patch(color='white', label='Standard Dev: '+str(std))
    med_patch = mpatches.Patch(color='white', label='Median: '+str(median))

    plt.legend(handles=[mean_patch,std_patch,med_patch])
    plt.xticks(x, keys, fontsize=8, rotation=75)
    plt.show()

def document_genre_distributions():
    NotImplemented

def document_subtypes_distributions():
    NotImplemented

def pca():
    NotImplemented


if __name__ == '__main__':
    print('starting document processing')
    overall_page_distribution()
    print('Starting ari calculation')
    ari_distribution()
    document_genre_distributions()
    document_subtypes_distributions()
    pca()
