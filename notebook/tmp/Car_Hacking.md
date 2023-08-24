■ Black Hat Car Hacking
Jailbreaking an Electric Vehicle in 2023 or What It Means to Hotwire Tesla's x86-Based Seat Heater

Tesla Model 3に対する攻撃。ICEに搭載のOSのルート権限を奪取して、本来であれば有料オプションである座席ヒート機能を不正に作動させることに成功

脅威アクター：車に物理的にアクセスできる人
攻撃対象ECU: Infortainment and Connectivity ECU (ICE)
当該ECUの特筆すべき特徴
    - FreeRTOS-based OSが搭載
    - 座席ヒート機能を含む、車の設定を平文で管理 
    - タッチスクリーンやナビゲーション、エンターテインメント・システム    
    
攻撃方法
    ルートFSにSSHのルートアカウントのパスワードもしくは鍵情報を追加して他PCからリモートログインすることでルート権限奪取する。
    本来、起動時に完全性チェックが走り、keyファイルへの書き換えを検知できるため、SSHの鍵情報などの書き換えは不可能。
    ただ、ICEに搭載のCPU (AMD v1)はフォルトインジェクション攻撃に脆弱で、成功すると完全性チェックを回避することができる。
    その結果、SSHによるリモートログインが成功してしまい、ルート権限が奪取されてしまう。

また、座席ヒート機能の不正有効化にも、fTPMにより保護されている、クレデンシャル情報を格納する領域にアクセスして、Tesla VPNに接続するための認証情報やドライバの個人情報を取得することに成功

感想: 
　- 平文ではなくハッシュ化して保存していればVPN情報が盗まれることはなかったと思われる。なので「多層防御 is とても 大事」
  - 本件に限らすテスラのハッキングとかではデフォでファーム解析などしていたので、やはりその技術が必要なのかなって気がした。（じゃないと、暗闇の中を進むようなことになってしまう・・・） 
　
　
https://i.blackhat.com/BH-US-23/Presentations/US-23-Werling-Jailbreaking-Teslas.pdf?_gl=1*1es83c5*_gcl_aw*R0NMLjE2OTA0NzE3MTQuQ2p3S0NBandxNGltQmhCUUVpd0E5TngxQmljU2xTR3hvOEpGNHMxRm5faFBUMzdidjI3ekt3REFFRHdSN1VVcVhqbm1WNV9COHBlcnhSb0NkYzRRQXZEX0J3RQ..*_gcl_au*MzA3NTI5Njg0LjE2ODcyNzQzNTc.*_ga*MTUzMTUwMTk5OS4xNjkwNDczMTgx*_ga_K4JK67TFYV*MTY5MjgwNTY1My4yNy4xLjE2OTI4MDU2NjEuMC4wLjA.&_ga=2.238880009.888904598.1692805653-1531501999.1690473181&_gac=1.158943688.1690471714.CjwKCAjwq4imBhBQEiwA9Nx1BicSlSGxo8JF4s1Fn_hPT37bv27zKwDAEDwR7UUqXjnmV5_B8perxRoCdc4QAvD_BwE


----------------------------------------------------------------------
