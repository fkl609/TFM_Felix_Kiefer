"""
Modelo Vanilla RNN a nivel de carácter.
Arquitectura adaptada para predicción simultánea (Multi-head): 12 notas de entrada -> 4 notas de salida directa.
"""
import numpy as np
import pickle
import matplotlib.pyplot as plt
import optuna
from optuna.samplers import TPESampler

# ==========================================
# 0. CONFIGURACIÓN DE REPRODUCIBILIDAD
# ==========================================
SEED = 31
np.random.seed(SEED)

# ==========================================
# 1. CARGA DE DATOS
# ==========================================
with open('../dataset/input.txt', 'r') as f:
    lines = f.readlines()
    
songs = [line.strip() for line in lines if line.strip()]
all_text = ''.join(songs)
chars = sorted(list(set(all_text))) 
data_size, vocab_size = len(all_text), len(chars)
char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }

# ==========================================
# 2. HIPERPARÁMETROS FIJOS
# ==========================================
seq_length = 12   # Longitud de la secuencia de contexto (entrada)
output_length = 4 # Número de notas a predecir simultáneamente

# ==========================================
# 3. FUNCIONES DEL MODELO
# ==========================================
def lossFun(inputs, targets, hprev, Wxh, Whh, Whys, bh, bys):
    xs, hs, ps = {}, {}, {}
    hs[-1] = np.copy(hprev)
    loss = 0
    
    # 1. FORWARD PASS (Contexto): Procesamiento de las 12 notas para construir el estado oculto
    for t in range(len(inputs)):
        xs[t] = np.zeros((vocab_size, 1)) 
        xs[t][inputs[t]] = 1
        hs[t] = np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t-1]) + bh) 
    
    h_final = hs[len(inputs)-1]

    # 2. PREDICCIÓN SIMULTÁNEA: Aplicación de las 4 cabezas (Whys) sobre el estado oculto final
    for i in range(output_length):
        y = np.dot(Whys[i], h_final) + bys[i]
        p = np.exp(y) / np.sum(np.exp(y))
        ps[i] = p
        loss += -np.log(p[targets[i], 0])
    
    # 3. BACKWARD PASS
    dWxh = np.zeros_like(Wxh)
    dWhh = np.zeros_like(Whh)
    dbh = np.zeros_like(bh)
    dWhys = [np.zeros_like(w) for w in Whys]
    dbys = [np.zeros_like(b) for b in bys]
    
    # El error se origina en las 4 cabezas simultáneas y se agrega en h_final
    dh = np.zeros_like(h_final)
    for i in range(output_length):
        dy = np.copy(ps[i])
        dy[targets[i]] -= 1
        
        dWhys[i] += np.dot(dy, h_final.T)
        dbys[i] += dy
        dh += np.dot(Whys[i].T, dy)
        
    dhnext = dh
    
    # Propagación del error acumulado hacia atrás a través de los pasos de contexto
    for t in reversed(range(len(inputs))):
        dhraw = (1 - hs[t] * hs[t]) * dhnext 
        dbh += dhraw
        dWxh += np.dot(dhraw, xs[t].T)
        dWhh += np.dot(dhraw, hs[t-1].T)
        dhnext = np.dot(Whh.T, dhraw)
        
    # Recorte de gradientes (clipping) para mitigar el problema de explosión de gradientes
    for dparam in [dWxh, dWhh, dbh] + dWhys + dbys:
        np.clip(dparam, -5, 5, out=dparam) 
        
    return loss, dWxh, dWhh, dWhys, dbh, dbys, h_final

# ==========================================
# 4. OPTIMIZACIÓN CON OPTUNA
# ==========================================
def objective(trial):
    hidden_size = trial.suggest_int('hidden_size', 50, 200, step=25) 
    # Exploración discreta de la tasa de aprendizaje para claridad en la memoria del TFM
    learning_rate = trial.suggest_categorical('learning_rate', [1e-3, 5e-3, 1e-2, 5e-2, 1e-1])
    batch_size = trial.suggest_categorical('batch_size', [15, 30, 60])
    epochs_trial = 50 

    # Inicialización de pesos y sesgos
    Wxh = np.random.randn(hidden_size, vocab_size)*0.01 
    Whh = np.random.randn(hidden_size, hidden_size)*0.01 
    bh = np.zeros((hidden_size, 1)) 
    
    Whys = [np.random.randn(vocab_size, hidden_size)*0.01 for _ in range(output_length)]
    bys = [np.zeros((vocab_size, 1)) for _ in range(output_length)]

    mWxh, mWhh, mbh = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(bh)
    mWhys = [np.zeros_like(w) for w in Whys]
    mbys = [np.zeros_like(b) for b in bys]
    
    Acc_dWxh, Acc_dWhh, Acc_dbh = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(bh)
    Acc_dWhys = [np.zeros_like(w) for w in Whys]
    Acc_dbys = [np.zeros_like(b) for b in bys]
    
    batch_counter = 0
    smooth_loss = 0

    for epoch in range(epochs_trial):
        epoch_loss = 0
        steps = 0
        
        for current_song in songs:
            hprev = np.zeros((hidden_size, 1))
            
            # Extracción iterativa de bloques (12 contexto + 4 predicción)
            for p in range(len(current_song) - seq_length - output_length + 1):
                inputs = [char_to_ix[ch] for ch in current_song[p : p+seq_length]]
                targets = [char_to_ix[ch] for ch in current_song[p+seq_length : p+seq_length+output_length]]

                loss, dWxh, dWhh, dWhys, dbh, dbys, hprev = lossFun(inputs, targets, hprev, Wxh, Whh, Whys, bh, bys)
                epoch_loss += loss
                steps += 1

                Acc_dWxh += dWxh
                Acc_dWhh += dWhh
                Acc_dbh += dbh
                for i in range(output_length):
                    Acc_dWhys[i] += dWhys[i]
                    Acc_dbys[i] += dbys[i]
                
                batch_counter += 1
                
                if batch_counter >= batch_size:
                    # Agrupación de parámetros y gradientes para actualización por lotes (Adagrad)
                    all_params = [Wxh, Whh, bh] + Whys + bys
                    all_grads = [Acc_dWxh, Acc_dWhh, Acc_dbh] + Acc_dWhys + Acc_dbys
                    all_mems = [mWxh, mWhh, mbh] + mWhys + mbys
                    
                    for i in range(len(all_params)):
                        dparam = all_grads[i] / batch_size
                        all_mems[i] += dparam * dparam
                        all_params[i] += -learning_rate * dparam / np.sqrt(all_mems[i] + 1e-8) 
                    
                    Acc_dWxh, Acc_dWhh, Acc_dbh = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(bh)
                    Acc_dWhys = [np.zeros_like(w) for w in Whys]
                    Acc_dbys = [np.zeros_like(b) for b in bys]
                    batch_counter = 0
        
        smooth_loss = epoch_loss / steps if steps > 0 else 0
        if np.isnan(smooth_loss) or smooth_loss > 100: 
            raise optuna.TrialPruned()
            
    return smooth_loss

# ==========================================
# 5. BUCLE PRINCIPAL (MAIN)
# ==========================================
if __name__ == "__main__":
    # Configuración del optimizador bayesiano con semilla fija
    sampler = TPESampler(seed=SEED)
    study = optuna.create_study(direction='minimize', sampler=sampler)
    study.optimize(objective, n_trials=15)

    best_hparams = study.best_params
    hidden_size = best_hparams['hidden_size'] 
    learning_rate = best_hparams['learning_rate']
    batch_size = best_hparams['batch_size']
    epochs_final = 500 

    print("--- OPTIMIZACIÓN FINALIZADA ---")
    print(f"Entrenando modelo final con -> H_Size: {hidden_size}, LR: {learning_rate}, Batch: {batch_size}")

    # Reinicio de semilla para garantizar condiciones idénticas en el entrenamiento final
    np.random.seed(SEED) 
    Wxh = np.random.randn(hidden_size, vocab_size)*0.01 
    Whh = np.random.randn(hidden_size, hidden_size)*0.01 
    bh = np.zeros((hidden_size, 1)) 
    Whys = [np.random.randn(vocab_size, hidden_size)*0.01 for _ in range(output_length)]
    bys = [np.zeros((vocab_size, 1)) for _ in range(output_length)]

    mWxh, mWhh, mbh = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(bh)
    mWhys = [np.zeros_like(w) for w in Whys]
    mbys = [np.zeros_like(b) for b in bys]

    Acc_dWxh, Acc_dWhh, Acc_dbh = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(bh)
    Acc_dWhys = [np.zeros_like(w) for w in Whys]
    Acc_dbys = [np.zeros_like(b) for b in bys]
    
    batch_counter = 0
    loss_history = []
    epoch_history = []

    for epoch in range(epochs_final):
        epoch_loss = 0
        steps = 0
        
        for current_song in songs:
            hprev = np.zeros((hidden_size, 1))

            for p in range(len(current_song) - seq_length - output_length + 1):
                inputs = [char_to_ix[ch] for ch in current_song[p : p+seq_length]]
                targets = [char_to_ix[ch] for ch in current_song[p+seq_length : p+seq_length+output_length]]

                loss, dWxh, dWhh, dWhys, dbh, dbys, hprev = lossFun(inputs, targets, hprev, Wxh, Whh, Whys, bh, bys)
                epoch_loss += loss
                steps += 1
                
                Acc_dWxh += dWxh
                Acc_dWhh += dWhh
                Acc_dbh += dbh
                for i in range(output_length):
                    Acc_dWhys[i] += dWhys[i]
                    Acc_dbys[i] += dbys[i]
                
                batch_counter += 1
                
                if batch_counter >= batch_size:
                    all_params = [Wxh, Whh, bh] + Whys + bys
                    all_grads = [Acc_dWxh, Acc_dWhh, Acc_dbh] + Acc_dWhys + Acc_dbys
                    all_mems = [mWxh, mWhh, mbh] + mWhys + mbys
                    
                    for i in range(len(all_params)):
                        dparam = all_grads[i] / batch_size
                        all_mems[i] += dparam * dparam
                        all_params[i] += -learning_rate * dparam / np.sqrt(all_mems[i] + 1e-8) 
                    
                    Acc_dWxh, Acc_dWhh, Acc_dbh = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(bh)
                    Acc_dWhys = [np.zeros_like(w) for w in Whys]
                    Acc_dbys = [np.zeros_like(b) for b in bys]
                    batch_counter = 0
        
        smooth_loss = epoch_loss / steps if steps > 0 else 0
        
        if epoch % 10 == 0:
            print('Epoch %d, loss: %f' % (epoch, smooth_loss)) 
            epoch_history.append(epoch)
            loss_history.append(smooth_loss)

    # Visualización y exportación
    plt.plot(epoch_history, loss_history)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    
    # Sustitución de puntos decimales para evitar conflictos en el sistema de archivos
    lr_str = str(learning_rate).replace('.', '') 
    plt.savefig('../results/plots/rnn_opt_%d_%s_%d.png' % (hidden_size, lr_str, batch_size)) 
    plt.show()

    # Estructuración de datos para serialización del modelo completo
    model_data = {
        'Wxh': Wxh,
        'Whh': Whh,
        'Whys': Whys, 
        'bh': bh,
        'bys': bys,
        'char_to_ix': char_to_ix,
        'ix_to_char': ix_to_char,
        'hidden_size': hidden_size,
        'vocab_size': vocab_size,
        'seq_length': seq_length, 
        'output_length': output_length,
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'epochs': epochs_final
    }

    with open('../models/rnn_model_opt_%d_%s_%d.pkl' % (hidden_size, lr_str, batch_size), 'wb') as f: 
        pickle.dump(model_data, f)