import librosa
import numpy as np

# Retorna a máscara relativa à variância
def var_trust_func(Y, numFrames=200):
    x = int(Y.shape[1]/numFrames)
    var_trust = np.ones(Y.shape)
    
    for i in range(0, Y.shape[1], numFrames):
        indice_start = i
        indice_stop = indice_start+numFrames

        Y_p = Y[:,indice_start:indice_stop]
        var_p = np.sqrt(Y_p.var(1))
        var_p = var_p.reshape(var_p.shape[0],1)   
        var_trust[:,indice_start:indice_stop] = var_p
        
    var_trust = var_trust / np.max(var_trust)
    return(var_trust)

# Expande a matriz de contraste para as mesmas dimensões da matriz espectral
def expand_contrast(contrast_p, shape, n_bands, deltaF):
    contrast = np.ones(shape)
    for i in range(0, n_bands+1):        
        if i == 0:
            indice_start = 0
        else:
            indice_start = deltaF * 2**(i-1)
        
        indice_stop = deltaF * 2**i - 1
        contrast[indice_start:indice_stop,:] = contrast_p[i,:]

    contrast[indice_stop+1:,:] = contrast_p[n_bands,:]    
    return(contrast)

# Retorna a máscara relativa ao contraste
def contrast_trust_func(Y, sr):
    n_bands = 8
    contrast_p = librosa.feature.spectral_contrast(S=Y, sr=sr, linear=True, n_bands=n_bands, fmin=64)
    deltaF = int(round(64/(sr/(2*Y.shape[0]))))
    contrast = expand_contrast(contrast_p, Y.shape, n_bands, deltaF)
    contrast /= np.max(contrast)
    return(contrast)

# Filtro 1: máscara de filtragem = máscara da variância .* máscara do contraste
def my_filter(y, sr):
    Y = librosa.stft(y, n_fft = 4096, hop_length = 512)
    Y_dB = librosa.amplitude_to_db(Y, ref=np.max)
       
    var_trust = var_trust_func(Y_dB)
    contrast = contrast_trust_func(np.abs(Y), sr)
    
    mask = np.multiply(contrast, var_trust)
    mask = mask / np.max(mask)    
    
    mag, phase = librosa.magphase(Y)
    newmag = np.multiply(mag, mask)
    Y_rec = np.multiply(newmag, np.exp(np.multiply(phase, (1j))))
    y_rec = librosa.istft(Y_rec, hop_length=512)

    return y_rec

# Filtro 2: filtragem prévia com máscara da variância, depois computação e filtragem da máscara de contraste
def my_filter2(y, sr):
    
    Y = librosa.stft(y, n_fft = 4096, hop_length = 512)
    Y_dB = librosa.amplitude_to_db(Y, ref=np.max)
       
    var_trust = var_trust_func(Y_dB)
    mask = var_trust
    
    mag, phase = librosa.magphase(Y)
    newmag = np.multiply(mag, mask)

    contrast = contrast_trust_func(np.abs(newmag), sr)
    mask = contrast
    
    x = 0.2 # "smoothing" para diminuir distorção do canto
    mask[mask < x] = x
    
    newnewmag = np.multiply(newmag, mask)
    Y_rec = np.multiply(newnewmag, np.exp(np.multiply(phase, (1j))))
    y_rec = librosa.istft(Y_rec, hop_length=512)
    return y_rec

# Filtro 3: filtro binário
def my_filter3(y, sr):
    
    Y = librosa.stft(y, n_fft = 4096, hop_length = 512)
    Y_dB = librosa.amplitude_to_db(Y, ref=np.max)
       
    var_trust = var_trust_func(Y_dB)
    contrast = contrast_trust_func(np.abs(Y), sr)
    
    mask = np.zeros(var_trust.shape)
    mask[np.logical_and(var_trust > 0.15, contrast > 0.2)] = 1
    
    mag, phase = librosa.magphase(Y)
    newmag = np.multiply(mag, mask)
    Y_rec = np.multiply(newmag, np.exp(np.multiply(phase, (1j))))
    
    y_rec = librosa.istft(Y_rec, hop_length=512)
    return y_rec