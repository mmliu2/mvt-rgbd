import matplotlib.pyplot as plt
import sys
import re


det_train_loss = [4.935275625, 5.9466931249999995, 3.4396006249999997, 2.829081875, 2.390564375, 2.481685, 2.322095, 1.977143125, 2.140895, 2.0088575, 2.01620875, 1.8862475, 1.92144625, 1.8208018750000001, 1.9040943750000001, 1.78237375, 1.6712056000000002, 1.92948125, 1.77917375, 1.64867, 1.544481875, 1.6388675, 1.627665, 1.580230625, 1.621875625]
det_train_f1 = [0.43653875, 0.53293125, 0.510479375, 0.577248125, 0.540185625, 0.570471875, 0.53249625, 0.59236125, 0.607114375, 0.607395, 0.63870625, 0.623368125, 0.646841875, 0.63264, 0.6514, 0.621670625, 0.6606048, 0.61206, 0.663041875, 0.68356375, 0.67390375, 0.6803525, 0.726651875, 0.700449375, 0.674739375]
det_test_loss = [2.53675, 2.093765, 2.023901875, 1.938004375, 1.8303375]
det_test_f1 = [0.53967, 0.538851875, 0.5864725, 0.638429375, 0.660201875]

log_path = 'output/logs/mobilevit_track_det-mvtdet_MED_256_128x1_depthtrack.log'

train_epochs = [*range(1, 26)]
test_epochs = [*range(5, 26, 5)]

with open(log_path, 'r') as f:
    lines = f.readlines()

    for line in lines:
        match = re.match(r'\[(\w+):\s*(\d+)', line)
        split, epoch = match.group(1), int(match.group(2))
        print(split, epoch)
        
        if split == 'train' and epoch == 1:
            print('reset')
            train_f1 = []
            test_f1 = []
            train_loss = []
            test_loss = []

        pattern = r'([\w\/]+):\s*([0-9\.]+)'
        matches = re.findall(pattern, line)
        metrics = {key: float(value) for key, value in matches}

        if split == 'train':
            if epoch == len(train_f1):
                train_f1[epoch-1] = metrics['depth/f1']
                train_loss[epoch-1] = metrics['depth/loss']
            else:
                train_f1.append(metrics['depth/f1'])
                train_loss.append(metrics['depth/loss'])
        elif split == 'val':
            if epoch == len(test_f1)*5:
                test_f1[epoch//5-1] = metrics['depth/f1']
                test_loss[epoch//5-1] = metrics['depth/loss']
            else:
                test_f1.append(metrics['depth/f1'])
                test_loss.append(metrics['depth/loss'])

mvt_train_loss = train_loss
mvt_train_f1 = train_f1
mvt_test_loss = test_loss
mvt_test_f1 = test_f1


# Font size variable
font_size = 16

fig, axes = plt.subplots(1, 4, figsize=(16, 3))  # 1 row, 4 columns

loss_train_color = '#7fc9ff'  # light blue
loss_test_color  = '#0073e6'  # blue
f1_train_color   = '#ffa9a9'  # light red / pink
f1_test_color    = '#e60000'  # red


# --- 1. DeT Loss ---
axes[0].plot(train_epochs, det_train_loss, marker='o', color=loss_train_color, label='Train')
axes[0].plot(test_epochs, det_test_loss, marker='o', color=loss_test_color, label='Test')
axes[0].set_xlabel("Epochs", fontsize=font_size)
axes[0].set_ylabel("Loss", fontsize=font_size)
axes[0].set_title("DeT Loss (MSE)", fontsize=font_size)
axes[0].legend(fontsize=font_size)
axes[0].grid(True)

# --- 2. DeT F1-Score ---
axes[1].plot(train_epochs, det_train_f1, marker='o', color=f1_train_color, label='Train')
axes[1].plot(test_epochs, det_test_f1, marker='o', color=f1_test_color, label='Test')
axes[1].set_xlabel("Epochs", fontsize=font_size)
axes[1].set_ylabel("F1", fontsize=font_size)
axes[1].set_title("DeT F1-Score", fontsize=font_size)
axes[1].legend(fontsize=font_size)
axes[1].grid(True)

# --- 3. Depth-Adapted MVT Loss ---
# axes[2].plot(train_epochs[:len(mvt_train_loss)], mvt_train_loss, marker='o', color=loss_train_color, label='Train')
# axes[2].plot(test_epochs[:len(mvt_test_loss)], mvt_test_loss, marker='o', color=loss_test_color, label='Test')
axes[2].plot(train_epochs[:len(mvt_train_loss)], mvt_train_loss, marker='o', color=loss_train_color)
axes[2].plot(test_epochs[:len(mvt_test_loss)], mvt_test_loss, marker='o', color=loss_test_color)
axes[2].set_xlabel("Epochs", fontsize=font_size)
axes[2].set_ylabel("Loss", fontsize=font_size)
axes[2].set_title("MVT-DeT Loss", fontsize=font_size)
# axes[2].legend(fontsize=font_size)
axes[2].grid(True)

# # --- 4. Depth-Adapted MVT F1-Score ---
# axes[3].plot(train_epochs[:len(mvt_train_f1)], mvt_train_f1, marker='o', color=f1_train_color, label='Train')
# axes[3].plot(test_epochs[:len(mvt_test_f1)], mvt_test_f1, marker='o', color=f1_test_color, label='Test')
axes[3].plot(train_epochs[:len(mvt_train_f1)], mvt_train_f1, marker='o', color=f1_train_color)
axes[3].plot(test_epochs[:len(mvt_test_f1)], mvt_test_f1, marker='o', color=f1_test_color)
axes[3].set_xlabel("Epochs", fontsize=font_size)
axes[3].set_ylabel("F1", fontsize=font_size)
axes[3].set_title("MVT-DeT F1-Score", fontsize=font_size)
# axes[3].legend(fontsize=font_size)
axes[3].grid(True)

# --- Single legend for the whole figure ---
lines_labels = [axes[0].get_lines()[0], axes[0].get_lines()[1]]  # first subplot's lines
labels = [line.get_label() for line in lines_labels]

# --- Increase horizontal spacing ---
fig.subplots_adjust(wspace=0.5, top=0.85, bottom=0.15)  # wspace controls space between subplots

plt.tight_layout()
plt.savefig('plotting/figs/med_det.png')
plt.show()