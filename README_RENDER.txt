COINSENTRA TRADING BRAIN V8 — RENDER DEPLOYMENT

FILES TO PUT IN THE ROOT OF ONE GITHUB REPOSITORY:
- app.py
- CoinSentra_Trading_Brain_V8.py
- requirements.txt
- render.yaml

DEPLOY:
1. Extract this ZIP on your phone.
2. Upload all four files to the ROOT of your GitHub repository (not inside another folder).
3. In Render, choose New + > Blueprint.
4. Connect the GitHub repository and deploy the blueprint.
5. The blueprint creates a paid Starter web service and a 1 GB persistent disk.
6. Wait for the deploy to finish. Open the service URL.
7. The browser asks for HTTP Basic Authentication:
   Username: coinsentra
   Password: the generated DASHBOARD_PASSWORD shown in the Render service Environment settings.

IMPORTANT:
- PAPER TRADING ONLY. No Bybit API keys are used and no real orders are placed.
- The 8x setting is simulated. TP and SL are 2% price moves.
- Scanner uses Bybit public market data for ETHUSDT, XRPUSDT, SOLUSDT and SUIUSDT.
- The $6 daily loss stop applies to simulated realized P&L.
- Keep the service running; a continuously scanning app needs an always-on paid service.
- Persistent disk stores the paper account state and log across deploys/restarts.
- Never add exchange API keys to this paper-only version.

If you previously created a Render service for an older version, deploy this as a new Blueprint or replace its repository contents with these files and check the service settings carefully.
