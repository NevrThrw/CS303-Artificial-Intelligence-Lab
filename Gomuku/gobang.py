import numpy as np
import time
import re
import random

color_black = -1
color_white = 1
color_none = 0

# 下完棋后棋型
pattern = [r'11111',  # 成五，胜利
           r'011110',  # 活四，必胜
           r'211110', r'11101', r'11011', r'10111', r'011112',  # 冲四
           r'001110', r'011100', r'010110', r'011010',  # 活三
           r'211100', r'001112', r'211010', r'210110', r'011012', r'010112', r'10011', r'10101', r'11001', r'2011102',
           # 眠三
           r'01100', r'00110', r'0010100', r'1001',  # 活二
           r'211000', r'210100', r'000112', r'001012', r'210010', r'010012', r'2010102', r'10001'  # 眠二
           ]

# 棋型得分
pattern_score = [300000,
                 100000,
                 30000, 30000, 30000, 30000, 30000,
                 15000, 15000, 15000, 15000,
                 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000,
                 3000, 3000, 3000, 3000,
                 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000
                 ]

class AI(object):

    def __init__(self, chessboard_size, color, time_out):
        self.candidate_list = []  # 落子位置的备选方案
        self.chessboard_size = chessboard_size  # 棋盘的大小
        self.color = color  # 我方落子的颜色
        self.time_out = time_out
        self.attack_num = 1.3
        self.defence_num = 1
        self.blank = 0  # 统计棋面上空位的个数
        self.enemy = 0
        self.my = 0

    def get_candidate_simple(self, chessboard):
        lis = []
        self.blank = 0
        for i in range(15):
            for j in range(15):
                if chessboard[i][j] == 0:
                    self.blank += 1
                    if self.check_neighbors(chessboard, i, j):
                        lis.append((i, j))
        return lis

    def check_neighbors(self, chessboard, x, y):  # 判断周围7*7的区域内是否有子，如果没有则不考虑在这落子
        for i in range(-3, 4):
            for j in range(-3, 4):
                if i == 0 and j == 0:
                    continue
                elif self.chessboard_size - 1 >= x + i >= 0 and self.chessboard_size - 1 >= y + j >= 0:  # 边界判定
                    if chessboard[x + i][y + j] != 0:
                        return True
        return False

    def count_score(self, chessboard, index, chessboard_size, color):  # 计算棋型的得分，选取以落子处为中心的9*9的区域进行遍历
        x = index[0]  # 行
        y = index[1]  # 列
        score_list = []
        lis = []

        # left to right
        left = '1'
        right = ''
        for i in range(1, 5):
            if y - i >= 0:  # 边界判定
                if chessboard[x][y - i] == (-color):  # 如果搜索到了对手的棋子，则停止搜索

                    left = '2' + left
                    break
                elif chessboard[x][y - i] == 0:
                    if len(left) > 1 and left[0] == '0' and left[1] == '0':  # 如果出现连续的三个空位，则停止搜索
                        left = '0' + left
                        break
                    else:
                        left = '0' + left
                else:
                    left = '1' + left
            else:
                left = '2' + left
                break

        for i in range(1, 5):
            if y + i <= chessboard_size - 1:
                if chessboard[x][y + i] == (-color):
                    right = right + '2'
                    break
                elif chessboard[x][y + i] == 0:
                    if len(right) > 1 and right[-1] == '0' and right[-2] == '0':
                        right = right + '0'
                        break
                    else:
                        right = right + '0'
                else:
                    right = right + '1'
            else:
                right = right + '-1'
                break
        lis.append(left + right)

        # up to down
        up = '1'
        down = ''
        for i in range(1, 5):
            if x - i >= 0:  # 边界判定
                if chessboard[x - i][y] == (-color):  # 如果搜索到了对手的棋子，则停止搜索
                    up = '2' + up
                    break
                elif chessboard[x - i][y] == 0:  # 如果出现连续的两个空位，则停止搜索
                    if len(up) > 1 and up[0] == '0' and up[1] == '0':
                        up = '0' + up
                        break
                    else:
                        up = '0' + up
                else:
                    up = '1' + up
            else:
                up = '2' + up
                break

        for i in range(1, 5):
            if x + i <= chessboard_size - 1:
                if chessboard[x + i][y] == (-color):
                    down = down + '2'
                    break
                elif chessboard[x + i][y] == 0:
                    if len(down) > 1 and down[-1] == '0' and down[-2] == '0':
                        down = down + '0'
                        break
                    else:
                        down = down + '0'
                else:
                    down = down + '1'
            else:
                down = down + '2'
                break
        lis.append(up + down)

        # left_up to right_down
        lu = '1'
        rd = ''
        for i in range(1, 5):
            if y - i >= 0 and x - i >= 0:
                if chessboard[x - i][y - i] == (-color):
                    lu = '2' + lu
                    break
                elif chessboard[x - i][y - i] == 0:
                    if len(lu) > 1 and lu[0] == '0' and lu[1] == '0':
                        lu = '0' + lu
                        break
                    else:
                        lu = '0' + lu
                else:
                    lu = '1' + lu
            else:
                lu = '2' + lu
                break

        for i in range(1, 5):
            if y + i <= chessboard_size - 1 and x + i <= chessboard_size - 1:
                if chessboard[x + i][y + i] == (-color):
                    rd = rd + '2'
                    break
                elif chessboard[x + i][y + i] == 0:

                    if len(rd) > 1 and rd[-1] == '0' and rd[-2] == '0':
                        rd = rd + '0'
                        break
                    else:
                        rd = rd + '0'
                else:
                    rd = rd + '1'
            else:
                rd = rd + '2'
                break

        lis.append(lu + rd)

        # right_up to left_down
        ld = '1'
        ru = ''
        for i in range(1, 5):
            if x - i >= 0 and y + i <= chessboard_size - 1:
                if chessboard[x - i][y + i] == (-color):
                    ru = ru + '2'
                    break
                elif chessboard[x - i][y + i] == 0:
                    if len(ru) > 1 and ru[-1] == '0' and ru[-2] == '0':
                        ru = ru + '0'
                        break
                    else:
                        ru = ru + '0'
                else:
                    ru = ru + '1'
            else:
                ru = ru + '2'
                break

        for i in range(1, 5):
            if x + i <= chessboard_size - 1 and y - i >= 0:
                if chessboard[x + i][y - i] == (-color):
                    ld = '2' + ld
                    break
                elif chessboard[x + i][y - i] == 0:
                    if len(ld) > 1 and ld[0] == '0' and ld[1] == '0':
                        ld = '0' + ld
                        break
                    else:
                        ld = '0' + ld
                else:
                    ld = '1' + ld
            else:
                ld = '2' + ld
                break

        lis.append(ld + ru)
        # 统计落子后形成的棋型
        special = {'five': 0, 'live_three': 0, 'cut_four': 0, 'live_two': 0, 'live_four': 0, 'cut_three': 0}
        score = 0
        for chess in lis:
            for pat in pattern:
                if re.search(pat, chess):
                    cost = pattern_score[pattern.index(pat)]
                    if cost == 300000:
                        special['five'] += 1
                    if cost == 100000:
                        special['live_four'] += 1
                    if cost == 15000:
                        special['live_three'] += 1
                    if cost == 30000:
                        special['cut_four'] += 1
                    if cost == 3000:
                        special['live_two'] += 1
                    if cost == 5000:
                        special['cut_three'] += 1
                    score += cost
                    score_list.append(cost)
                    break
        return score, special

    def count_stone_around(self, chessboard, index, color):
        x, y = index[0], index[1]
        self.enemy = 0
        self.my = 0
        for i in range(-1, 2):
            for j in range(-1, 2):
                if i == 0 and j == 0:
                    continue
                elif 14 >= x + i >= 0 and 14 >= y + j >= 0:  # 边界判定
                    if chessboard[x + i][y + j] == -color:
                        self.enemy += 1
                    else:
                        self.my += 1

    def go(self, chessboard):
        print(self.color)
        self.candidate_list.clear()
        lis = self.get_candidate_simple(chessboard)
        best = -1
        if self.blank == 0:  # 棋盘满了
            return
        if self.blank == 15 * 15:
            self.candidate_list.append([7, 7])
        if self.blank <= 145:  # 如果双方下了40手仍未获胜
            self.attack_num = 1.2
            self.defence_num = 1.1
        for i in lis:
            value = 0
            attack, a = self.count_score(chessboard, i, 15, self.color)
            defence, d = self.count_score(chessboard, i, 15, -self.color)
            self.count_stone_around(chessboard, i, self.color)
            exist = False
            # 攻击特殊棋形有冲四活三，双活三
            if a['live_three'] > 0 and a['cut_four'] > 0:
                attack += 25000  # 冲四活三
            if a['live_three'] == 2:
                attack += 40000  # 双活三
            if a['live_two'] > 2:
                attack += 7000  # 三活二
            if a['live_three'] > 0 and a['cut_three'] > 1:
                attack += 7000  # 活三双眠三
            if a['live_three'] > 0 and a['live_two'] > 0 and a['cut_three'] > 0:
                attack += 10000
            # 防守特殊棋型
            if d['live_three'] > 0 and d['cut_four'] > 0:
                defence += 25000  # 冲四活三
                exist = True
            if d['live_three'] == 2:
                defence += 40000  # 双活三
            if d['cut_four'] == 2:
                exist = True
            if d['cut_four'] > 0 and d['live_two'] > 1:  # 冲四双活二
                exist = True
            if d['cut_four'] > 0 and d['cut_three'] > 1:  # 冲四双眠三
                exist = True
            if d['live_two'] > 2:
                defence += 7000
            if d['live_three'] > 0 and d['cut_three'] > 1:
                attack += 7000
            if d['live_three'] > 0 and d['live_two'] > 0 and d['cut_three'] > 0:
                defence += 32000
            if self.color == color_black:  # 如果我方执黑，进攻为主
                value = attack * self.attack_num + defence * self.defence_num + self.my * 20
                attack += self.my * 20
            if self.color == color_white:  # 如果我方执白，防守为主
                value = attack * self.defence_num + defence * self.attack_num + self.enemy * 20
                defence += self.enemy * 20
            # print(i, attack, defence, value)
            # print(a)
            # print(d)
            # 概率忽略对手眠三可能形成的冲四
            ram = random.random()
            if not exist and d['cut_four'] == 1:
                if self.color == color_black and ram < 0.8:
                    value -= 50000 * self.defence_num
                elif self.color == color_white and ram < 0.65:
                    value -= 50000 * self.attack_num

            if value > best:  # 最优选择是得分最高
                self.candidate_list.append([i[0], i[1]])
                best = value
        print(self.candidate_list)


if __name__ == '__main__':
    pass
