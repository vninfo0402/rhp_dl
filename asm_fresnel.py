import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift

# 파라미터 설정
wavelength = 532e-9  # 파장 (532 nm)
k = 2 * np.pi / wavelength  # 파수
z = 0.01  # 전파 거리 (10 mm)
N = 512  # CGH 해상도
L = 0.01  # 영역 크기 (10 mm)

# 공간 좌표
dx = L / N
x = np.linspace(-L/2, L/2-dx, N)  # 수정된 공간 좌표
y = np.linspace(-L/2, L/2-dx, N)
X, Y = np.meshgrid(x, y)

# 주파수 좌표
dfx = 1 / L
fx = np.fft.fftfreq(N, dx)  # 수정된 주파수 좌표
fy = np.fft.fftfreq(N, dx)
FX, FY = np.meshgrid(fx, fy)

# 입력 파면 (가우시안 빔)
sigma = L/8
U0 = np.exp(-(X**2 + Y**2)/(2*sigma**2))

try:
    # ASM 전달 함수
    kx = 2 * np.pi * FX
    ky = 2 * np.pi * FY
    kz = np.sqrt(k**2 - kx**2 - ky**2 + 0j)
    
    # 에바네센트 파 필터링
    prop_mask = np.real(kz) > 0
    H_ASM = np.exp(1j * kz * z) * prop_mask

    # ASM 적용
    U1 = ifft2(fftshift(fft2(U0)) * H_ASM)  # fftshift 위치 수정

    # 강도와 위상 계산
    intensity_ASM = np.abs(U1)**2
    phase_ASM = np.angle(U1)

    # ASM 결과 출력
    plt.figure(figsize=(12, 5))
    
    plt.subplot(121)
    plt.imshow(intensity_ASM, cmap='viridis', extent=[-L/2*1000, L/2*1000, -L/2*1000, L/2*1000])
    plt.colorbar(label='Intensity (a.u.)')
    plt.xlabel('x (mm)')
    plt.ylabel('y (mm)')
    plt.title("ASM Intensity")
    
    plt.subplot(122)
    plt.imshow(phase_ASM, cmap='twilight', extent=[-L/2*1000, L/2*1000, -L/2*1000, L/2*1000])
    plt.colorbar(label='Phase (rad)')
    plt.xlabel('x (mm)')
    plt.ylabel('y (mm)')
    plt.title("ASM Phase")
    
    plt.tight_layout()
    plt.show()

    # Fresnel 전달 함수
    H_Fresnel = np.exp(1j * k * z) * np.exp(-1j * np.pi * wavelength * z * (FX**2 + FY**2))

    # Fresnel 적용
    U1_Fresnel = ifft2(fftshift(fft2(U0)) * H_Fresnel)  # fftshift 위치 수정

    # 강도와 위상 계산
    intensity_Fresnel = np.abs(U1_Fresnel)**2
    phase_Fresnel = np.angle(U1_Fresnel)

    # Fresnel 결과 출력
    plt.figure(figsize=(12, 5))
    
    plt.subplot(121)
    plt.imshow(intensity_Fresnel, cmap='viridis', extent=[-L/2*1000, L/2*1000, -L/2*1000, L/2*1000])
    plt.colorbar(label='Intensity (a.u.)')
    plt.xlabel('x (mm)')
    plt.ylabel('y (mm)')
    plt.title("Fresnel Intensity")
    
    plt.subplot(122)
    plt.imshow(phase_Fresnel, cmap='twilight', extent=[-L/2*1000, L/2*1000, -L/2*1000, L/2*1000])
    plt.colorbar(label='Phase (rad)')
    plt.xlabel('x (mm)')
    plt.ylabel('y (mm)')
    plt.title("Fresnel Phase")
    
    plt.tight_layout()
    plt.show()

except Exception as e:
    print(f"오류가 발생했습니다: {str(e)}")
    raise