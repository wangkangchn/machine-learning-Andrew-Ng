"""
            K-Means 划分算法相关函数

"""
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import image as pimg        # mpimg 用于读取图片

cnames = {
'aliceblue':            '#F0F8FF',
'antiquewhite':         '#FAEBD7',
'aqua':                 '#00FFFF',
'aquamarine':           '#7FFFD4',
'azure':                '#F0FFFF',
'beige':                '#F5F5DC',
'bisque':               '#FFE4C4',
'black':                '#000000',
'blanchedalmond':       '#FFEBCD',
'blue':                 '#0000FF',
'blueviolet':           '#8A2BE2',
'brown':                '#A52A2A',
'burlywood':            '#DEB887',
'cadetblue':            '#5F9EA0',
'chartreuse':           '#7FFF00',
'chocolate':            '#D2691E',
'coral':                '#FF7F50',
'cornflowerblue':       '#6495ED',
'cornsilk':             '#FFF8DC',
'crimson':              '#DC143C',
'cyan':                 '#00FFFF',
'darkblue':             '#00008B',
'darkcyan':             '#008B8B',
'darkgoldenrod':        '#B8860B',
'darkgray':             '#A9A9A9',
'darkgreen':            '#006400',
'darkkhaki':            '#BDB76B',
'darkmagenta':          '#8B008B',
'darkolivegreen':       '#556B2F',
'darkorange':           '#FF8C00',
'darkorchid':           '#9932CC',
'darkred':              '#8B0000',
'darksalmon':           '#E9967A',
'darkseagreen':         '#8FBC8F',
'darkslateblue':        '#483D8B',
'darkslategray':        '#2F4F4F',
'darkturquoise':        '#00CED1',
'darkviolet':           '#9400D3',
'deeppink':             '#FF1493',
'deepskyblue':          '#00BFFF',
'dimgray':              '#696969',
'dodgerblue':           '#1E90FF',
'firebrick':            '#B22222',
'floralwhite':          '#FFFAF0',
'forestgreen':          '#228B22',
'fuchsia':              '#FF00FF',
'gainsboro':            '#DCDCDC',
'ghostwhite':           '#F8F8FF',
'gold':                 '#FFD700',
'goldenrod':            '#DAA520',
'gray':                 '#808080',
'green':                '#008000',
'greenyellow':          '#ADFF2F',
'honeydew':             '#F0FFF0',
'hotpink':              '#FF69B4',
'indianred':            '#CD5C5C',
'indigo':               '#4B0082',
'ivory':                '#FFFFF0',
'khaki':                '#F0E68C',
'lavender':             '#E6E6FA',
'lavenderblush':        '#FFF0F5',
'lawngreen':            '#7CFC00',
'lemonchiffon':         '#FFFACD',
'lightblue':            '#ADD8E6',
'lightcoral':           '#F08080',
'lightcyan':            '#E0FFFF',
'lightgoldenrodyellow': '#FAFAD2',
'lightgreen':           '#90EE90',
'lightgray':            '#D3D3D3',
'lightpink':            '#FFB6C1',
'lightsalmon':          '#FFA07A',
'lightseagreen':        '#20B2AA',
'lightskyblue':         '#87CEFA',
'lightslategray':       '#778899',
'lightsteelblue':       '#B0C4DE',
'lightyellow':          '#FFFFE0',
'lime':                 '#00FF00',
'limegreen':            '#32CD32',
'linen':                '#FAF0E6',
'magenta':              '#FF00FF',
'maroon':               '#800000',
'mediumaquamarine':     '#66CDAA',
'mediumblue':           '#0000CD',
'mediumorchid':         '#BA55D3',
'mediumpurple':         '#9370DB',
'mediumseagreen':       '#3CB371',
'mediumslateblue':      '#7B68EE',
'mediumspringgreen':    '#00FA9A',
'mediumturquoise':      '#48D1CC',
'mediumvioletred':      '#C71585',
'midnightblue':         '#191970',
'mintcream':            '#F5FFFA',
'mistyrose':            '#FFE4E1',
'moccasin':             '#FFE4B5',
'navajowhite':          '#FFDEAD',
'navy':                 '#000080',
'oldlace':              '#FDF5E6',
'olive':                '#808000',
'olivedrab':            '#6B8E23',
'orange':               '#FFA500',
'orangered':            '#FF4500',
'orchid':               '#DA70D6',
'palegoldenrod':        '#EEE8AA',
'palegreen':            '#98FB98',
'paleturquoise':        '#AFEEEE',
'palevioletred':        '#DB7093',
'papayawhip':           '#FFEFD5',
'peachpuff':            '#FFDAB9',
'peru':                 '#CD853F',
'pink':                 '#FFC0CB',
'plum':                 '#DDA0DD',
'powderblue':           '#B0E0E6',
'purple':               '#800080',
'red':                  '#FF0000',
'rosybrown':            '#BC8F8F',
'royalblue':            '#4169E1',
'saddlebrown':          '#8B4513',
'salmon':               '#FA8072',
'sandybrown':           '#FAA460',
'seagreen':             '#2E8B57',
'seashell':             '#FFF5EE',
'sienna':               '#A0522D',
'silver':               '#C0C0C0',
'skyblue':              '#87CEEB',
'slateblue':            '#6A5ACD',
'slategray':            '#708090',
'snow':                 '#FFFAFA',
'springgreen':          '#00FF7F',
'steelblue':            '#4682B4',
'tan':                  '#D2B48C',
'teal':                 '#008080',
'thistle':              '#D8BFD8',
'tomato':               '#FF6347',
'turquoise':            '#40E0D0',
'violet':               '#EE82EE',
'wheat':                '#F5DEB3',
'white':                '#FFFFFF',
'whitesmoke':           '#F5F5F5',
'yellow':               '#FFFF00',
'yellowgreen':          '#9ACD32'}

def find_closest_centroids (X, centroids):
    """ 依据聚簇中心, 划分样本 """
    K = centroids.shape[0]
    idx = np.zeros((X.shape[0], 1))

    # 样本于每个聚簇中心的距离平方值
    distance = np.zeros((K, 1))
    # 循环每个样本设置所属的中心
    for i in range(X.shape[0]):
        # 获取每行的最小值下标, 即为样本所属的聚簇中心
        idx[i] = np.argmin(np.sum(np.power((X[i, :] - centroids), 2), axis=1), axis=0)
    return idx

def compute_centroids(X, idx, K):
    """ 移动聚簇中心到样本的均值处 """
    # 聚簇中心
    centroids = np.zeros((K, X.shape[1]))

    # 移动聚簇中心
    for i in range(K):
        # 被赋值0是因为出现了没有单独的聚簇点8
        centroids[i, :] = np.mean(X[np.where(idx==i)[0], :], axis=0)

    return centroids

def plot_data_points(X, idx, K):
    """ 绘制样本点 """
    for k, c in zip(range(K), cnames.values()):
        plt.plot(X[np.where(idx==k)[0], 0], X[np.where(idx==k)[0], 1], 'o', color=c)

def draw_line(p1, p2):
    """ 绘制聚簇中心的移动轨迹 """
    plt.plot([p1[0], p2[0]], [p1[1], p2[1]])

def plot_progress_kMeans(X, centroids, previous, idx, K, i):
    """ 绘制K-Means过程 """

    # 绘制样本点
    plot_data_points(X, idx, K);

    # 将聚簇中心绘制成为黑色的x
    plt.plot(centroids[:, 0], centroids[:, 1], 'xk', MarkerSize=10, LineWidth=3)

    # 用线条绘制中心走过的路程
    for j in range(centroids.shape[0]):
        draw_line(centroids[j, :], previous[j, :])

    plt.title('Iteration number {}'.format(i))

def run_kMeans(X, centroids, max_iters, plot_progress=False):
    """ 运行K-Means算法 """
    # 初始化参数
    m, n = X.shape
    K = centroids.shape[0]
    idx = np.zeros((m, 1))
    previous_centroids = centroids

    if plot_progress:
        # 进行动态绘图
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        # 开启交互模式直接绘图, 不需要show
        plt.ion()

    for i in range(max_iters):
        print('K-Means iteration {}/{}...'.format(i+1, max_iters))
        # 绘制K-Means的过程
        if plot_progress:
            plot_progress_kMeans(X, centroids, previous_centroids, idx, K, i)
            previous_centroids = centroids
            # 暂停一秒
            plt.pause(1)
        # 分配
        idx = find_closest_centroids(X, centroids)
        # 移动
        centroids = compute_centroids(X, idx, K)

    if plot_progress:
        # 关闭交互模式
        plt.ioff()

    return centroids, idx

def kMeans_init_centroids(X, K):
    """ 初始化聚类中心 """

    centroids = np.zeros((K, X.shape[1]))

    # 获取一个随机序列
    randidx = np.arange(X.shape[0])
    np.random.shuffle(randidx)

    # 取随机序列的前K个数作为初始聚簇中心
    centroids = X[randidx[:K], :]

    return centroids




