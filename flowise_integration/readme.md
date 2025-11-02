# Librechat flowise integration
ref.: https://docs.flowiseai.com/using-flowise/prediction
## Phase 1
1. Create proxy adapter to convert flowise /api/v1/prediction to openai completion compatible payload.
2. Supply the /api/v1/prediction coverted endpoint as standard custom endpoints to librechat in librechat.yaml
## Phase 2
Add a flowise model managent method without having to restart server.


