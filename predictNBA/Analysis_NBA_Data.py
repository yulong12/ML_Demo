import pandas
import math
import random
import csv
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression


class NBA():
	def __init__(self, Mstats, Ostats, Tstats, results_data, schedules):
		self.base_elo = 1600
		self.team_elos = {}
		self.teams_stats = {}
		self.X = []
		self.y = []
		self.model = LogisticRegression()
		self.Mstats = Mstats
		self.Ostats = Ostats
		self.Tstats = Tstats
		self.results_data = results_data
		self.schedules = schedules
		self.prediction_results = []
	# 运行用
	def run(self):
		# 数据初始化
		self.teams_stats = self.initialize_data(self.Mstats, self.Ostats, self.Tstats)
		self.X, self.y = self.build_DataSet(self.results_data)
		self.train_model()
		print('[INFO]:starting predict...')
		for index, row in self.schedules.iterrows():
			team1 = row['Vteam']
			team2 = row['Hteam']
			pred = self.predict(team1, team2)
			if pred[0][0] > 0.5:
				self.prediction_results.append([team1, team2, pred[0][0]])
			else:
				self.prediction_results.append([team2, team1, 1-pred[0][0]])
		return self.prediction_results
	# 数据初始化
	def initialize_data(self, Mstats, Ostats, Tstats):
		print('[INFO]:Initialize Train Data...')
		# 去除一些不需要的数据
		initial_Mstats = Mstats.drop(['Rk', 'Arena'], axis=1)
		initial_Ostats = Ostats.drop(['Rk', 'G', 'MP'], axis=1)
		initial_Tstats = Tstats.drop(['Rk', 'G', 'MP'], axis=1)
		# 将三个表格通过Team属性列进行连接
		temp = pandas.merge(initial_Mstats, initial_Ostats, how='left', on='Team')
		all_stats = pandas.merge(temp, initial_Tstats, how='left', on='Team')
		return all_stats.set_index('Team', inplace=False, drop=True)
	# 建立数据集
	def build_DataSet(self, results_data):
		print('[INFO]:Building DataSet...')
		X = []
		y = []
		for index, row in results_data.iterrows():
			Wteam = row['WTeam']
			Lteam = row['LTeam']
			# 获取elo值
			Wteam_elo = self.get_team_elo(Wteam)
			Lteam_elo = self.get_team_elo(Lteam)
			# 给主场比赛的队伍加上100的elo值
			if row['WLoc'] == 'H':
				Wteam_elo += 100
			else:
				Lteam_elo += 100
			# elo作为评价每个队伍的第一个特征值
			Wteam_features = [Wteam_elo]
			Lteam_features = [Lteam_elo]
			# 添加其他统计信息
			for key, value in self.teams_stats.loc[Wteam].iteritems():
				Wteam_features.append(value)
			for key, value in self.teams_stats.loc[Lteam].iteritems():
				Lteam_features.append(value)
			# 两支队伍的特征值随机分配
			if random.random() > 0.5:
				X.append(Wteam_features + Lteam_features)
				y.append(0)
			else:
				X.append(Lteam_features + Wteam_features)
				y.append(1)
			# 根据比赛数据更新队伍Elo值
			new_winner_rank, new_loser_rank = self.calc_elo(Wteam, Lteam)
			self.team_elos[Wteam] = new_winner_rank
			self.team_elos[Lteam] = new_loser_rank
		return np.nan_to_num(X), np.array(y)
	# 计算每支队伍的Elo值
	def calc_elo(self, win_team, lose_team):
		winner_rank = self.get_team_elo(win_team)
		loser_rank = self.get_team_elo(lose_team)
		rank_diff = winner_rank - loser_rank
		exp = (rank_diff * -1) / 400
		# Winner对loser的胜率期望
		E_w_to_l = 1 / (1 + math.pow(10, exp))
		# 根据rank级别修改K值
		if winner_rank < 2100:
			K = 32
		elif winner_rank >= 2100 and winner_rank < 2400:
			K = 24
		else:
			K = 16
		new_winner_rank = round(winner_rank + (K * (1 - E_w_to_l)))
		new_rank_diff = new_winner_rank - winner_rank
		new_loser_rank = loser_rank - new_rank_diff
		return new_winner_rank, new_loser_rank
	# 获取每支队伍的Elo Score等级分
	def get_team_elo(self, team):
		try:
			return self.team_elos[team]
		except:
			# 初始elo值
			self.team_elos[team] = self.base_elo
			return self.team_elos[team]
	# 训练网络模型
	def train_model(self):
		print('[INFO]:Trainning model...')
		self.model.fit(self.X, self.y)
		# 10折交叉验证计算训练正确率
		print(cross_val_score(self.model, self.X, self.y, cv=10, scoring='accuracy', n_jobs=-1).mean())
	# 用于预测
	def predict(self, team1, team2):
		features = []
		# team1为客场队伍
		features.append(self.get_team_elo(team1))
		for key, value in self.teams_stats.loc[team1].iteritems():
			features.append(value)
		# team2为主场队伍
		features.append(self.get_team_elo(team2)+100)
		for key, value in self.teams_stats.loc[team2].iteritems():
			features.append(value)
		features = np.nan_to_num(features)
		return self.model.predict_proba([features])





if __name__ == '__main__':
	# 综合统计数据
	Mis_Stats = pandas.read_csv('./data/16-17Miscellaneous_Stats.csv')
	# 每支队伍的对手平均每场比赛的表现统计
	Opp_Stats = pandas.read_csv('./data/16-17Opponent_Per_Game_Stats.csv')
	# 每支队伍平均每场比赛的表现统计
	Tea_Stats = pandas.read_csv('./data/16-17Team_Per_Game_Stats.csv')
	# 16-17年每场比赛的数据集
	results_data = pandas.read_csv('./data/2016-2017_results.csv')
	# 17-18年比赛安排
	schedule16_17 = pandas.read_csv('./data/17-18Schedule.csv')
	pred_results = NBA(Mis_Stats, Opp_Stats, Tea_Stats, results_data, schedule16_17).run()
	print('[INFO]:start saving pred results...')
	with open('17-18Result.csv', 'w') as f:
		writer = csv.writer(f)
		writer.writerow(['winner', 'loser', 'probability'])
		writer.writerows(pred_results)
		f.close()
	print('[INFO]:All things done...')