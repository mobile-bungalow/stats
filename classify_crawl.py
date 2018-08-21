import os
import readchar
import pickle
if __name__ == '__main__':
    print("Beggining classifier")
    streaming = []
    geo = []
    social_media = []
    search_engines = []
    banking = []
    retail = []
    class_dict = {'streaming':streaming,
                    'geo':geo,
                    "social_media":social_media,
                    'search_engines':search_engines,
                    'banking':banking}
    
    file_list = os.listdir('./web_crawl')

    char_to_name = {'a':'streaming','s':'geo','d': "social_media",'f':'search_engines','g':'banking','h':'retail'}

    while(file_list != []):
        print("Classify "+ file_list[0]+" A[streaming], S[geo], D[social media], f[search engines], G[banking],H[retail], U[undo] R[none]")
        char = readchar.readkey()
        website = file_list.pop()
        file_list = file_list[1:]
        prev_char = char
        if char in 'asdfg':
            class_dict[char_to_name[char]].append(website)
        elif char == 'u':
            file_list.append(class_dict[prev_char].pop())
        elif char == 'r':
            pass
        if char == 't':
            file_list = []
    
    print(class_dict)

    f = open('dict_pickle.pickle','wb')

    pickle.dump(class_dict,f)

    f.close()



