import matplotlib.pyplot as plt


x1 = [11.78374749, 52.32412371, 21.28915499, 17.16403442, 3.832738707, 19.35720544, 17.68114851, 19.34277595, 8.925428207, 57.64815823]
x2 = [41.42700688, 19.41051398, 71.04736755, 42.57980328, 46.18944927, 29.67684275, 33.43132408, 18.13791593, 10.8692973, 20.6442862]
x3 = [10.68377515, 8.394019144, 7.708401938, 7.342050046, 39.24488088, 6.61658844, 4.423900465, 14.48528178, 18.77682793, 18.593797]
x4 = [18.72161318, 51.44114725, 34.00924879, 30.7623723, 19.12656468, 6.058489663, 29.72848007, 12.02125145, 23.71902766, 40.69555131]
x5 = [8.935098276, 13.82428042, 13.64057695, 6.04950626, 13.46735177, 8.42594075, 20.78843413, 25.70149096, 24.13934979, 10.9854888]
x6 = [13.80088866, 25.20524431, 12.87238679, 16.13744672, 38.4314883, 11.76249594, 38.85211181, 5.902910673, 23.61670366, 13.16900235]
x7 = [8.206284122, 9.530929957, 27.23679503, 36.40684319, 52.90285199, 10.7456855, 45.35978457, 10.33077879, 9.806547269, 20.06284095]
x8 = [12.0742251, 24.50710201, 18.76869049, 14.23832832, 25.30238019, 21.44589564, 46.1597232, 15.80201255, 8.376910748, 11.30026572]


# fig1, ax = plt.subplots()
# ax.boxplot((x1, x2, x3, x4))
# ax.set_xticklabels(["transverse", "diagonal-1", "longitudinal", "diagonal-2"], fontsize=16)
# plt.xlabel('Probe Position', fontsize=20)
# plt.ylabel('RMSE', fontsize=20)
# # plt.show()
# # fig1.savefig("boxplot1" + ".png")

# fig2, ax = plt.subplots()
# ax.boxplot((x5, x6, x7, x8))
# ax.set_xticklabels(["transverse", "diagonal-1", "longitudinal", "diagonal-2"], fontsize=16)
# plt.xlabel('Probe Position', fontsize=20)
# plt.ylabel('RMSE', fontsize=20)
# # plt.show()
# # fig2.savefig("boxplot2" + ".png")


fig, ax = plt.subplots()
ax.boxplot((x1, x2, x3, x4, x5, x6, x7, x8))
ax.set_xticklabels(["transverse\n(slow)", "transverse\n(fast)", "diagonal-1\n(slow)", "diagonal-1\n(fast)", "longitudinal\n(slow)", "longitudinal\n(fast)", "diagonal-2\n(slow)", "diagonal-2\n(fast)"], fontsize=8)
plt.xlabel('Probe Position', fontsize=18)
plt.ylabel('RMSE', fontsize=18)
plt.tight_layout()
# plt.show()
fig.savefig("boxplot" + ".png")