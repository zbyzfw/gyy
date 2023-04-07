# -*- mode: python ; coding: utf-8 -*-

block_cipher = None
path = 'C:\\用户\\AA\\miniconda3\\envs\\paddle0629'
a = Analysis(
    # 入口程序位置
    ['detec_list.py'],
    # python包搜索路径，这里填写一个conda环境目录，一个当前目录
    pathex=[path,'.'],
    binaries=[],
    # 复制项目中的模型文件，配置文件及测试用图片
    datas=[('models_v1','models_v1'),('meter.json','.'),('ppocr','ppocr'),
    (path+'\\Lib\\site-packages\\paddle','paddle'),(path+'\\Lib\\site-packages\\skimage','skimage')],
    # 需要手动导入的库
    hiddenimports=['pywt'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # 不参与打包的库
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
    # 是否显示控制台
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
