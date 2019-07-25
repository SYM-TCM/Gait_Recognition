import cv2
import numpy as np
from PIL import Image
import glob
import xlwt
import xlrd
import os
import pandas as pd
from pandas import DataFrame
from sklearn.cluster import KMeans

workbook = xlwt.Workbook(encoding='utf-8',style_compression=0)
sheet = workbook.add_sheet('sym1',cell_overwrite_ok=True)

workbook1 = xlwt.Workbook(encoding='utf-8', style_compression=0)
sheet1 = workbook1.add_sheet('sym2', cell_overwrite_ok=True)

workbook2 = xlwt.Workbook(encoding='utf-8',style_compression=0)
sheet2 = workbook2.add_sheet('sym3',cell_overwrite_ok=True)


Change_name='sym'


def ImageCutting(index,P):
    """
    第一步：将原图进行初步剪切，将高大于430部分剪切掉，然后在进行闭运算
    再寻找图片的最大轮廓面积并剪切

    """
    for k in range(1, P):
        src1 = cv2.imread("E://"+str(Change_name)+"//" + str(index) + "//Color_bg" + str(k) + ".png")
        gray = cv2.cvtColor(src1, cv2.COLOR_BGR2GRAY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))  # 循环次
        dilated = cv2.dilate(gray, kernel)  # 膨胀
        eroded = cv2.erode(dilated, kernel)  # 腐蚀

        contours, hierarchy = cv2.findContours(eroded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        hw_list = []
        x_di = {}
        y_di = {}
        h_di = {}
        w_di = {}
        HW = 0
        max = 0
        # 下面这段代码就是轮廓外围的边框，也可以调节边框的颜色及宽度，（153，153，0），1
        for i in range(0, len(contours)):
            x, y, w, h = cv2.boundingRect(contours[i])
            HW = h * w
            x_di[HW] = x
            y_di[HW] = y
            h_di[HW] = h
            w_di[HW] = w
            hw_list.append(HW)
            max = np.max(hw_list)
            # 下面这代码会修改gray原图，有必要须备份gray
            # 此时的gray的边缘轮廓会加上方框，并替换原来的gary。
            cv2.rectangle(gray, (x, y), (x + w, y + h), (153, 153, 0), 1)
            # cv2.imshow("aaa",gray)
            # cv2.waitKey(0)

        # 剪切  [起始点纵坐标：结束点纵坐标，起始点横坐标：结束点横坐标]

        cropImg = eroded[y:h + y, x:x + w]
        cv2.imwrite("E://"+str(Change_name)+"//" + str(index) + "//1_" + str(k) + ".png", cropImg)

        """
        第二步：将上述的人体轮廓移动大小为1200x1200的图片中央

        """
        image = Image.new('RGB', (1200, 1200), (0, 0, 0))
        image1 = Image.open("E://"+str(Change_name)+"//" + str(index) + "//1_" + str(k) + ".png")
        width1 = image1.size[0]
        height1 = image1.size[1]

        box1 = (0, 0, width1, height1)

        cropIM = image1.crop(box1)  # 复制的图片
        # 横坐标，纵坐标
        image.paste(cropIM, (500, 400))  # 粘贴的位置和对应图片
        # image.show(image)

        image = image.convert('L')  # 此时必须转换为 mode = ’L‘，或者一开始就转换成，
        # 不然，在matlab找质心代码的bwlabel(),会报错，格式必须为L--灰度图

        image.save("E://"+str(Change_name)+"//" + str(index) + "//2_" + str(k) + ".png")

        """"
        第三步：将上诉1200x1200图片的找出质心所在，然后利用质心进行分割
        """

        # print(M)
        img1 = cv2.imread("E://"+str(Change_name)+"//" + str(index) + "//2_" + str(k) + ".png", 0)
        ret, thresh = cv2.threshold(img1, 127, 255, 0)
        contours, hierarchy = cv2.findContours(thresh, 1, 2)
        cnt = contours[0]
        M = cv2.moments(cnt)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        #print(cx, cy)
        sheet.write(k, 0, cx)
        sheet.write(k, 1, cy)
        workbook.save("E://"+str(Change_name)+"//" + str(index) + "//No.1.xls")

        """
          上面完成了各图片的质心寻找，并写入excel表中
          分界线
          下面将利用质心，通过质心，进行定点位置分割，以质心为中心分割！

          """

        data = xlrd.open_workbook("E:/"+str(Change_name)+"/" + str(index) + "/No.1.xls")
        table = data.sheets()[0]
        col = int(table.cell_value(k, 1))  # row,col 获取指定单元格的数据-从0开始，所以需要-1
        row = int(table.cell_value(k, 0))

        # print(111111)
        # print(row,col)
        cropped = img1[col - 400:col + 400, row - 300:row + 300]  # 纵坐标，横坐标 matlab 的x对应高，即纵坐标，
        # 高400，宽300，不同图片来源不同，可根据实际图片大小变化
        cv2.imwrite("E://"+str(Change_name)+"//" + str(index) + "//3_" + str(k) + ".png", cropped)


def GEI(index,P):

    """
    第四步：
    上面完成了利用质心坐标进行分割，
    分界线
    下面将完成质心分割后最终图片，进行步态能量图的生成
    """
    imgs_list = []
    for i in range(1, P):
        imgs_list.append(cv2.imread("E://"+str(Change_name)+"//"+str(index)+"//3_"+ str(i)+".png") / 255)  # 利用列表存储图片，for循环，,除以255是归一化处理
        # 步态能量图中的计算公式将一个步态内的图像叠加在一起后除以图片数目
    GEI = imgs_list[0]  # 第一张图
    for i in imgs_list[1:]:  # 依次循环叠加图片
        GEI += i
    GEI = GEI / len(imgs_list)  # 除以图片数量

        # cv2.imshow('GEI', GEI)
    GEI1 = GEI * 255  # 保存图片乘以255.否则全黑
    #cv2.imwrite("E://7.18//"+str(index)+"//GEI_"+str(index)+".png", GEI1)
    cv2.imwrite("E://"+str(Change_name)+"//GEI//GEI_"+str(index)+".png",GEI1)

    ret, thresh = cv2.threshold(GEI1, 160, 255, cv2.THRESH_BINARY)

    cv2.imwrite("E://"+str(Change_name)+"//"+str(index)+"//GEI_Thresh_"+str(index)+".png",thresh)


def GEI_grouping():
    path_file_number = glob.glob("E:\\"+str(Change_name)+"\\GEI\\*.png")  # 获取当前文件夹目标文件的个数
    P = len(path_file_number) + 1  # +1的原因是因为是python-for q in range(1,new_name)少1
    imgs_list = []

    for i in range(1, P):
        imgs_list.append(
            cv2.imread("E://"+str(Change_name)+"//GEI//GEI_" + str(i) + ".png") / 255)  # 利用列表存储图片，for循环，,除以255是归一化处理
        # 步态能量图中的计算公式将一个步态内的图像叠加在一起后除以图片数目
    GEI = imgs_list[0]  # 第一张图
    for i in imgs_list[1:]:  # 依次循环叠加图片
        GEI += i
    GEI = GEI / len(imgs_list)  # 除以图片数量

    # cv2.imshow('GEI', GEI)
    GEI1 = GEI * 255  # 保存图片乘以255.否则全黑
    # cv2.imwrite("E://7.18//"+str(index)+"//GEI_"+str(index)+".png", GEI1)
    cv2.imwrite("E://"+str(Change_name)+"//GEI//Goal_Gei.png", GEI1)

    ret, thresh = cv2.threshold(GEI1, 160, 255, cv2.THRESH_BINARY)

    cv2.imwrite("E://"+str(Change_name)+"//GEI//GEI.png", thresh)


def Proportion(new_name,P):
    # for循环15张图，从第11张到25张结束！


    for j in range(1, new_name):
        for i in range(1, P):
            # 1.导入原图及对比图
            original_img = cv2.imread("E://"+str(Change_name)+"//GEI//GEI.png", 0)
            compare_img = cv2.imread("E:/"+str(Change_name)+"/" + str(j) + "/3_" + str(i) + ".png", 0)

            # 2.得到行，列
            height = original_img.shape[0]
            width = original_img.shape[1]

            # 3.创建原图存储的中间列表及结果列表
            list_org_middle = []
            list_org_result = []

            # 4.遍历行，列-图
            for row in range(height):
                for col in range(width):
                    if original_img[row][col] == 255:  # 如果坐标[row][col]的像素点为255
                        # 中间列表-一维列表！存储[row,col]
                        list_org_middle.append(row)
                        list_org_middle.append(col)
                        # 结果列表- 二维列表！ 存储，每一个符合条件的像素点
                        list_org_result.append(list_org_middle)
                        list_org_middle = []  # 将中间列表存储到结果列表后，清空中间列表，为下个像素点存储作准备！

            # 打印输出，符合条件的像素点的个数
            # print(len(list2))

            list_repeat_middle = []
            list_repeat_result = []

            for row in range(height):
                for col in range(width):
                    # 如果母图该坐标的像素点与子图该坐标的像素点=255，将该坐标存储起来
                    if (original_img[row][col] == 255) and (compare_img[row][col] == 255):
                        list_repeat_middle.append(row)
                        list_repeat_middle.append(col)
                        list_repeat_result.append(list_repeat_middle)
                        list_repeat_middle = []

            # print(len(list4))

            result = ("%.4f" % (len(list_repeat_result) / len(list_org_result)))

            # print(result)
            """
            sheet.write(1,0,'hjf-lame-1')
            sheet.write(2,0,'hjf-normal-1')
            sheet.write(3,0,'hjf-normal-2')
            sheet.write(4,0,'hjf-normal-3')
            sheet.write(5,0,'hjf-normal-4')
            sheet.write(6,0,'sym-lame-1')
            sheet.write(7,0,'sym-normal-1')
            sheet.write(8,0,'sym-normal-2')
            sheet.write(9,0,'sym-normal-3')
            sheet.write(10,0,'sym-normal-4')
            sheet.write(11,0,'zml-lame-1')
            sheet.write(12,0,'zml-normal-1')
            sheet.write(13,0,'zml-normal-2')
            sheet.write(14,0,'zml-normal-3')
            sheet.write(15,0,'zml-normal-4')
            """

            # sheet.write(0,0,'normal-1')
            # sheet.write(1,0,'normal-2')
            # sheet.write(2,0,'lame-2')
            sheet1.write(j, i - 1, result)

            workbook1.save("E://"+str(Change_name)+"//cluster_data.xls")
            #cluster_data='E://7.18//cluster_data.xls'
        # print('result:{:.2%}'.format(len(list_repeat_result)/len(list_org_result)))
        # print("坐标：row "+str(row)+" col  " +str(col))


def Kmeans(new_name):
    # 1.读取文件
    datafile = "E:\\"+str(Change_name)+"\\cluster_data.xls"
    #datafile = cluster_data  # 文件所在位置，u为防止路径中有中文名称，此处没有，可以省略
    # outfile = 'E:\\5.7_out.xls'  # 设置输出文件的位置
    data = pd.read_excel(datafile)  # datafile是excel文件，所以用read_excel,如果是csv文件则用read_csv
    d = DataFrame(data)
    d.head()

    # 2. 聚类
    mod = KMeans(n_clusters=2, n_jobs=4, max_iter=500)  # 聚成3类数据,并发数为4，最大循环次数为500
    mod.fit_predict(d)  # y_pred表示聚类的结果

    # 聚成3类数据，统计每个聚类下的数据量，并且求出他们的中心
    r1 = pd.Series(mod.labels_).value_counts()
    r2 = pd.DataFrame(mod.cluster_centers_)
    r = pd.concat([r2, r1], axis=1)
    r.columns = list(d.columns) + [u'类别数目']
    # print(r)
    # 给每一条数据标注上被分为哪一类
    r = pd.concat([d, pd.Series(mod.labels_, index=d.index)], axis=1)
    r.columns = list(d.columns) + [u'聚类类别']
    # print((r[u'聚类类别']))
    # print('聚类类别')
    # print(r.iat[0,7])
    # print(r.iat[1,7])
    # print(r.iat[2,7])
    sheet2.write(0, 0, u'聚类类别')
    for index in range(1,new_name):
        sheet2.write(index, 0, int(r.iat[index-1, P-1]))

    #sheet2.write(3, 0, int(r.iat[1, P-1]))
    #sheet2.write(4, 0, int(r.iat[2, P-1]))
    #sheet2.write(5, 0, int(r.iat[3, P-1]))
    #sheet2.write(6, 0, int(r.iat[4, P-1]))
    #sheet2.write(7, 0, int(r.iat[5, P - 1]))
    #sheet2.write(8, 0, int(r.iat[6, P - 1]))
    #sheet2.write(9, 0, int(r.iat[7, P - 1]))
    #sheet2.write(10, 0, int(r.iat[8, P - 1]))
    #sheet2.write(11, 0, int(r.iat[9, P - 1]))
    #sheet2.write(12, 0, int(r.iat[10, P - 1]))
    #sheet2.write(13, 0, int(r.iat[11, P - 1]))

    #sheet2.write(14, 0, int(r.iat[12, P - 1]))
    #sheet2.write(15, 0, int(r.iat[13, P - 1]))
    # 具体分类的情况写在execl中，也可以打印输出！
    workbook2.save("E://"+str(Change_name)+"//Kmeans_Result.xls")




#主函数从这里执行

if __name__=='__main__':
    """
    下面两个部分分别为自动获取文件的数量，方便后续的遍历循环
    """
    path_file_number = glob.glob("E:\\"+str(Change_name)+"\\*")  # 获取当前文件夹目标文件的个数
    new_name = len(path_file_number) + 1  # +1的原因是因为是python-for q in range(1,new_name)少1   new_name = len(path_file_number) 因为多了GEI文件夹
    path_file_number = glob.glob("E:\\"+str(Change_name)+"\\1\\*.png")  # 获取当前文件夹目标文件的个数
    P = len(path_file_number) + 1  # +1的原因是因为是python-for q in range(1,new_name)少1
    os.mkdir("E:\\"+str(Change_name)+"\\GEI")
    for index in range(1,new_name):
        ImageCutting(index,P)
        GEI(index,P)
    GEI_grouping()
    Proportion(new_name,P)
    Kmeans(new_name)



