# NEXOR — Application staff locale

Dashboard web local (100% réseau local, aucune dépendance internet) qui relie
NEXOR (robot) au staff du restaurant : commandes en direct, statut du robot,
alertes, et position en salle.

```
application/
├── ws/
│   ├── main.py          ← serveur FastAPI + WebSocket (à lancer sur le Pi 5 ou un PC du resto)
│   ├── db.py             ← persistance SQLite : commandes, workflow, historique, timeline
│   ├── menu.json          ← menu du restaurant (catégories/plats/prix) — à adapter
│   ├── simulate.py       ← génère des événements fictifs pour tester sans robot
│   └── requirements.txt
├── ui/                     ← dashboard STAFF (cuisine/salle)
│   ├── index.html
│   ├── style.css
│   └── app.js
└── client_screen/          ← écran CLIENT (monté sur le robot)
    ├── index.html
    ├── style.css
    └── app.js
```

## Workflow des commandes (nouveau)

Les commandes ne sont plus juste "en cours / servie" : elles suivent un vrai
cycle de vie, persisté en SQLite (`ws/nexor_staff.db`, créé
automatiquement au premier lancement) :

```
confirmee → en_preparation → prete → servie → payee
       ↓              ↓
    annulee        annulee
```

- Chaque transition est horodatée dans `order_events` → c'est ce qui
  alimente la **timeline** (clique sur une commande dans le dashboard pour
  la voir).
- L'onglet **Historique** (`GET /api/orders/history`) montre toutes les
  commandes, y compris payées/annulées — l'onglet **En direct** ne montre
  que les commandes actives.
- Les transitions invalides (ex. sauter direct à "payée") sont rejetées
  par le serveur (`400 Bad Request`).
- Les données survivent à un redémarrage du serveur (testé).

## Paiement

Une commande **servie** peut être encaissée via `POST /api/orders/{id}/pay` :

```json
{ "method": "especes", "amount_paid": 20.0 }   // calcule la monnaie automatiquement
{ "method": "carte" }                            // pas de montant requis
```

Le serveur rejette : un encaissement avant que la commande soit "servie",
un montant en espèces insuffisant, et une méthode inconnue — tout est
validé côté serveur (testé), pas seulement côté interface.

Dans le dashboard, cliquer sur "Encaisser" ouvre une petite modale : choix
espèces/carte, saisie du montant reçu avec calcul de la monnaie en direct.

`GET /api/stats/today` inclut maintenant la répartition espèces/carte du
jour civil local (`encaissements`), base pour la future page Statistiques.

## 1. Lancer le serveur

```bash
cd application/ws
python -m pip install -r requirements.txt
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1
```

Pour un accès depuis un autre appareil, définis d'abord le secret partagé
(dans le terminal qui lance `application` **et** dans celui qui lance
`communication`) :

```powershell
$env:STAFF_APP_TOKEN = "remplace-par-un-secret-long"
```

Sous Linux/macOS : `export STAFF_APP_TOKEN="remplace-par-un-secret-long"`.

Le dashboard est servi automatiquement à `http://<IP_DU_PI>:8000/`.
Le staff ouvre juste cette adresse dans le navigateur de sa tablette/PC,
connectés au même WiFi local que le robot — aucune installation.

Pour trouver l'IP du Pi sur le réseau : `hostname -I`

## 2. Tester sans robot (démo)

```bash
# terminal 1
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1
# terminal 2
python simulate.py
```

Ouvre `http://localhost:8000/` : tu verras des commandes, une position de
robot qui bouge, et des alertes apparaître aléatoirement.

## 3. Brancher le vrai pipeline

### a) Communication

L'intégration est déjà réalisée dans `communication/nlp+llm/staff_delivery.py` :
file prioritaire, outbox persistante, rejeu idempotent des commandes et
corrélation des paiements par `order_id`. Ne rajoute pas d'appel `requests`
direct dans `dialog_manager.py`.

Configure uniquement `STAFF_APP_URL` et, si les processus ne tournent pas sur
la même machine, le même `STAFF_APP_TOKEN` dans les deux terminaux.

### b) Statut robot + position — côté ROS2

Depuis n'importe quel nœud ROS2 (navigation, state_machine…), pousse un HTTP
POST à chaque changement de statut / pose. Exemple minimal avec un
subscriber sur la pose et sur un topic d'état :

```python
import requests

STAFF_APP_URL = "http://localhost:8000"

def on_pose(msg):  # callback souscrivant à ta pose robot (ex: /amcl_pose ou /odom)
    # convertis tes coordonnées monde en 0-1 selon les dimensions de ta salle
    x_norm = (msg.pose.pose.position.x - X_MIN) / (X_MAX - X_MIN)
    y_norm = (msg.pose.pose.position.y - Y_MIN) / (Y_MAX - Y_MIN)
    requests.post(f"{STAFF_APP_URL}/api/events/robot_position",
                   json={"x": x_norm, "y": y_norm, "theta": yaw}, timeout=1)

def on_state_change(new_state):  # depuis ton state_machine.py (vision)
    mapping = {
        "CHERCHE_TABLE": ("en_route", ""),
        "ASSIS":         ("attente", ""),
        "BLOQUE":        ("bloque", "Obstacle détecté"),
        # ...
    }
    status, msg = mapping.get(new_state, ("ok", ""))
    requests.post(f"{STAFF_APP_URL}/api/events/robot_status",
                   json={"status": status, "message": msg}, timeout=1)
```

Adapte `TABLES` dans `application/ui/app.js` (coordonnées normalisées 0–1) aux
positions réelles des tables dans ta salle, pour que le plan corresponde à
la vraie disposition.

## Écran client (nouveau)

Interface destinée au client, à afficher sur l'écran monté sur le robot :
`http://<IP_DU_PI>:8000/client/`

- **Orbe vocal animé** : change de couleur/rythme selon l'état de la conversation
  (idle / listening / processing / speaking), avec un petit égaliseur animé
  pendant que le robot parle.
- **Transcription en temps réel** : affiche ce que le robot a compris pendant
  qu'il écoute/traite — le client peut vérifier qu'il a été bien compris.
- **Panier live** : les articles s'ajoutent visuellement au fur et à mesure
  de la commande, avant même la confirmation.
- **Icônes de langue** (FR/EN/عربي) qui s'allument selon la langue détectée.
- **Menu** : bouton "Voir le menu" (plein écran) + QR code généré localement
  (aucune dépendance internet) pointant vers `/menu`, une page HTML simple
  consultable depuis le téléphone du client sur le même WiFi.
- **Bouton "Appeler le staff"** : envoie directement une alerte au dashboard
  staff existant (réutilise `/api/events/robot_status`).

### Connexion à `dialog_manager.py`

Elle est déjà active via `staff_app_client.py` et `staff_delivery.py`. Les états
de dialogue, le brouillon, la commande confirmée et l'intention de paiement
passent tous par la même file robuste. Le backend ignore également le brouillon
identique que `communication` peut repousser juste après une confirmation.

### Adapter le menu

Édite `application/ws/menu.json` avec le vrai menu NEXOR (catégories, plats, prix,
description optionnelle). Mets aussi à jour `menu_url` avec l'IP réelle du
Pi sur le réseau du restaurant (`hostname -I`), sinon le QR code pointera
vers une IP de test qui ne fonctionnera pas depuis le téléphone du client.
Si le resto n'a pas de WiFi ouvert aux clients, remplace `menu_url` par
`"/menu"` tout court : le QR affichera alors un lien relatif, utile
seulement en test local, mais au moins la page `/menu` reste consultable
directement depuis l'écran du robot via le bouton "Voir le menu".

## 4. Sécurité réseau

Le CORS inter-origines est désactivé. Les appels locaux restent autorisés sans
configuration. Pour un accès depuis une tablette, un autre PC ou un robot
distant, définis le même secret `STAFF_APP_TOKEN` dans `application` et
`communication`, puis ouvre les interfaces avec le secret dans le fragment URL :

```text
http://<IP>:8000/#token=<SECRET>
http://<IP>:8000/client/?table=T4#token=<SECRET>
```

Le fragment n'est pas envoyé dans les requêtes HTTP et l'interface le retire
immédiatement de la barre d'adresse. `NEXOR_ALLOW_INSECURE_REMOTE=1` ne doit être
utilisé que pour une démonstration sur un réseau réellement isolé.
