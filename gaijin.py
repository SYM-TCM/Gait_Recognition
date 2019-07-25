import cv2
import os
import glob
import random

Change_name= 'sym'
list_num = []
list1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # 一组图片有10张，不可能都有效

def ScreenImage():
    """
    第一部分:  将去掉左边框以及右边框的图像和无人空白部分
    """
    for i in range(0,10):
        right = 0
        left = 0
        Mid = 0
        img1 = cv2.imread("E://"+str(Change_name)+"//"+str(name)+"//Color_bg"+str(i)+".png")
        img = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)#这一步很有必要，将24位深度的图像转成8位图像，方便后面
        height = img.shape[0]
        width = img.shape[1]


        for j in range(height):
            for k in range(width):
                if (340>j>140):
                    if(k==0):
                        if (img[j,0]==255).any():
                             left = left + 1
                    elif (k==639):
                         if  (img[j,639]==255).any():
                             right  = right + 1

                    elif j==240:
                        if  (img[240,k]==255).any():
                             Mid = Mid + 1
        #print("right:",right)
        #print("left:",left)
        #print("Mid",Mid)
        #print("\n")
        if(right>0):
            os.remove("E://"+str(Change_name)+"//"+str(name)+"//Color_bg"+str(i)+".png")
        elif(left>0):
            os.remove("E://"+str(Change_name)+"//"+str(name)+"//Color_bg"+str(i)+".png")
        elif(Mid==0):
            os.remove("E://"+str(Change_name)+"//"+str(name)+"//Color_bg" + str(i) + ".png")

def Rename():
    """
    第二部分:将上述删除掉剩余的图像--再重新排并重新排序命名
    """
    filepath = "E:\\"+str(Change_name)+"\\"+str(name)+""   #文件夹
    filelist =os.listdir(filepath)   #读入文件夹位置

    filenumber =0  #有必要性，列表都是从0开始的，而且python的变量必须赋值为0先

    for data in filelist:
        oldname =filepath+os.sep+filelist[filenumber]    #原来的文件名
        newname =filepath+os.sep+'Color_bg'+str(filenumber+1)+'.png'  #新的文件名
        os.rename(oldname,newname)   #将新的名字重新代替原来的文件名
        filenumber+=1            #注意，python无法使用i++ ，只能使用i+=1

def Srceen_Again():
    """
    第三部分:将剩下的图片进行剪切去除下面的脚步隐藏部分
    """
    list=[]
    path_file_number =glob.glob("E:\\"+str(Change_name)+"\\"+str(name)+"\\*.png")  #获取当前文件夹目标文件的个数
    new_name = len(path_file_number)+1         #+1的原因是因为是python-for q in range(1,new_name)少1

    #下面重新遍历剩下的图像进行剪切操作并重新保存新的图像
    for q in range(1,new_name):
        img1 = cv2.imread("E://"+str(Change_name)+"//" + str(name) + "//Color_bg" + str(q) + ".png")
        img = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) #同理
        height = img.shape[0]
        width = img.shape[1]

        for row in range(height):
            for col in range(width):
                if col==639:
                    if (img[row,639]==255).any():
                        list.append(row)
                        #遍历整个图像，判断width=639，也就是最后一列时，像素点等于255的row的位置加入list中，
                        #然后没循环每一张图片当width等于639，第一次出现255像素点的row作为剪切阈值点。



        goal_img = img[0:list[0]-25,0:640]#根据上述所找到的阈值点进行剪切
        cv2.imwrite("E://"+str(Change_name)+"//"+str(name)+"//Color_bg"+str(q)+".png",goal_img)
        list = []   #每剪完一张图像都重新将list置空

def FileNumber(list_num,new_name):

    """

    第四步：读取目标文件夹里的数量并存储在列表中
    print语句为调试而用
    7为总共多少组+1

    """

    for name in range(1, new_name):
        path_file_number = glob.glob("E:\\"+str(Change_name)+"\\" + str(name) + "\\*.png")
        num = len(path_file_number)
        list_num.append(num)

def DeleteImage(list1,list_num,new_name):
    """
    第五步：关键一步
    首先、判断某组是否要删除图片的，条件即是图片数量大于4
    其次、符合条件要删除的利用random 随机数函数从默认的list中，截取第二张到倒数第二张之间的范围，抽取的数量为某组图片数量-最小值
    最后、将抽取的图片给删除了

    """
    for name in range(1,new_name):
        if list_num[name-1]>min(list_num):
                p= random.sample(list1[1:list_num[name-1]-1],list_num[name-1]-min(list_num))
                for k in range(len(p)):
                    os.remove("E://"+str(Change_name)+"//"+str(name)+"//Color_bg"+str(p[k])+".png")


"""
第六步：删除之后，需要再次重新排序


for name in  range(1,7):
    filepath = "E:\\"+str(name)+""   #文件夹
    filelist =os.listdir(filepath)   #读入文件夹位置

    filenumber =0  #有必要性，列表都是从0开始的，而且python的变量必须赋值为0先

    for data in filelist:
        oldname =filepath+os.sep+filelist[filenumber]    #原来的文件名
        newname =filepath+os.sep+'Color_bg'+str(filenumber+1)+'.png'  #新的文件名
        os.rename(oldname,newname)   #将新的名字重新代替原来的文件名
        filenumber+=1            #注意，python无法使用i++ ，只能使用i+=1

"""

if __name__=='__main__':

    path_file_number =glob.glob("E:\\"+str(Change_name)+"\\*")  #获取当前文件夹目标文件的个数
    new_name = len(path_file_number)+1         #+1的原因是因为是python-for q in range(1,new_name)少1

    for name in range(1, new_name):

        listname = []
        ScreenImage()
        Rename()
        Srceen_Again()


    FileNumber(list_num,new_name)
    DeleteImage(list1,list_num,new_name)
    for name in range(1, new_name):
        Rename()

























