import preprocess as pre
import eval 


if __name__ == "__main__":
    pre.stopwordslist("./bin/stopwords.txt")
    text = {
        "text1" : "这句话呢，其实都是告诉你游戏规则，他就看你能不能看到他这个给你的规定了。",
        "text2" : "或者说你骂人一个游戏，它上面会有一个游戏的一个，这个攻略对不对？",
    }
    nn = eval.simChCheck()
    result = nn.forward(text["text1"],text["text2"])
    print(result)

"""
{
    "text1": "这句话呢，其实都是告诉你游戏规则，他就看你能不能看到他这个给你的规定了。", 
    "text2": "或者说你骂人一个游戏，它上面会有一个游戏的一个，这个攻略对不对？", 
    "similarity": 0.13900171220302582
}
"""