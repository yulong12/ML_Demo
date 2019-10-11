import jieba  
txt = open("./news.txt", encoding="utf-8").read()  
#加载停用词表  
stopwords = [line.strip() for line in open("CS.txt",encoding="utf-8").readlines()]  
# 加载特定词
specWords = [line.strip() for line in open("SpectWords.txt",encoding="utf-8").readlines()] 
words  = jieba.lcut(txt)  
counts = {}  
for word in words:  
    #不在停用词表中  
    if word not in stopwords:  
        #不统计字数为一的词  
        if len(word) == 1:  
            continue  
        elif word in specWords:  
            counts[word] = counts.get(word,0) + 1  
items = list(counts.items())  
items.sort(key=lambda x:x[1], reverse=True)   
File = open("result.txt",'a+')
for i in range(len(counts)):  
    word, count = items[i]  
    File.write(word+'\t')
    File.write(str(count)+'\n')
    File.flush()
    print ("{:<10}{:>7}".format(word, count))

