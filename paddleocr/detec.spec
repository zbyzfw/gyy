# -*- mode: python ; coding: utf-8 -*-


block_cipher = None

python_path = 'C:\用户\AA\miniconda3\envs\paddle0629'

a = Analysis(
    ['detec_list_web.py'],
    pathex=['C:\用户\AA\miniconda3\envs\paddle0629','.'],
    binaries=[],
    datas=[('models_v1','models_v1'),('ppocr','ppocr'),('meter.json','.')],
    hiddenimports=['pywt','paddle','skimage','ppocr'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='detec',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='detec_out',
)
