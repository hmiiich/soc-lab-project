"""
========================================================================
 MODELE DE DETECTION DE FRAUDE SUR TRANSACTIONS BANCAIRES
 py -3.12 fraud_detection_model.py data/mon_dataset.csv
========================================================================
Combine :
  1) Des regles metier explicites (une par cas d'usage demande)
  2) Un modele ML non supervise (Isolation Forest) qui apprend le
     comportement "normal" et detecte les ecarts
  3) Un modele ML supervise (Random Forest) si des labels historiques
     (Flag_Fraude_Potentiel) sont disponibles en nombre suffisant
  4) Un score de fusion qui reduit les faux positifs en exigeant une
     convergence de plusieurs signaux avant de declencher une alerte

Usage :
    python fraud_detection_model.py mon_dataset.csv

Sortie :
    - un CSV enrichi avec, pour CHAQUE transaction, le verdict
      (Normal / Anomalie) de chaque cas d'usage + un verdict final
    - un resume statistique en console
========================================================================
"""

import sys
import warnings
import logging
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import json

# Import optionnel de yaml (non obligatoire pour le fonctionnement)
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    warnings.warn("Module pyyaml non installé. Utilisation des valeurs par défaut. Installez-le avec: pip install pyyaml")



warnings.filterwarnings("ignore")

# Configuration du logging par défaut (avant chargement config)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fraud_detection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# CHARGEMENT DE LA CONFIGURATION
# ------------------------------------------------------------------
def charger_config(config_path: str = "config_fraude.yaml") -> Dict:
    """
    Charge la configuration depuis un fichier YAML.
    Si le fichier n'existe pas ou yaml n'est pas disponible, utilise les valeurs par défaut.
    """
    if not YAML_AVAILABLE:
        logger.warning("Module yaml non disponible. Utilisation des valeurs par défaut.")
        return {}
    
    config_path_obj = Path(config_path)
    
    if config_path_obj.exists():
        try:
            with open(config_path_obj, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration chargée depuis {config_path}")
            return config
        except Exception as e:
            logger.warning(f"Erreur lors du chargement de la configuration: {e}. Utilisation des valeurs par défaut.")
    
    logger.warning(f"Fichier de configuration non trouvé: {config_path}. Utilisation des valeurs par défaut.")
    return {}

# Charger la configuration
CONFIG = charger_config()

# Reconfigurer le logging si des paramètres sont fournis dans le fichier de config
if CONFIG.get('logging'):
    log_config = CONFIG.get('logging', {})
    log_level = getattr(logging, log_config.get('niveau', 'INFO'))
    log_file = log_config.get('fichier', 'fraud_detection.log')
    log_format = log_config.get('format', '%(asctime)s - %(levelname)s - %(message)s')
    
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# PARAMETRES / SEUILS (chargés depuis config_fraude.yaml ou valeurs par défaut)
# ------------------------------------------------------------------
seuils_config = CONFIG.get('seuils', {})
SEUIL_Z_SCORE_MONTANT = seuils_config.get('z_score_montant', 3.0)
SEUIL_VITESSE_SECONDES = seuils_config.get('vitesse_secondes', 60)
SEUIL_NB_BENEFICIAIRES_HEURE = seuils_config.get('nb_beneficiaires_heure', 3)
SEUIL_DISTANCE_KM_INCOHERENTE = seuils_config.get('distance_km_incoherente', 500)
SEUIL_VITESSE_KMH_IMPOSSIBLE = seuils_config.get('vitesse_kmh_impossible', 800)
SEUIL_HEURE_RARE_PCT = seuils_config.get('heure_rare_pct', 0.05)
SEUIL_DELAI_AJOUT_TRANSACTION = seuils_config.get('delai_ajout_transaction', 600)
SEUIL_NB_TRANS_FENETRE = seuils_config.get('nb_trans_fenetre', 5)
SEUIL_FENETRE_TEMPS = seuils_config.get('fenetre_temps', 120)

PAYS_A_RISQUE = set(CONFIG.get('pays_a_risque', ["Nigeria", "Panama", "Iran", "Corée du Nord", "Syrie", "Russie"]))

ml_config = CONFIG.get('ml', {})
CONTAMINATION_ISOLATION_FOREST = ml_config.get('contamination_isolation_forest', 0.05)

# ------------------------------------------------------------------
# MAPPING FLEXIBLE DES COLONNES (pour différents formats de données)
# ------------------------------------------------------------------
COLONNE_MAPPING = {
    # Colonnes obligatoires avec alternatives
    'Client_ID': ['client_id', 'ClientID', 'customer_id', 'CustomerID', 'id_client', 'ID'],
    'Date_Transaction': ['date_transaction', 'transaction_date', 'Date', 'date', 'timestamp'],
    'Montant': ['montant', 'amount', 'Amount', 'value', 'montant_eur', 'montant_usd'],
    
    # Colonnes optionnelles avec alternatives
    'Heure_Transaction': ['heure_transaction', 'time', 'Time', 'heure', 'transaction_time'],
    'Distance_Derniere_Transaction_KM': ['distance_km', 'distance', 'Distance', 'last_distance_km'],
    'Vitesse_Transactions_Secondes': ['vitesse_sec', 'velocity_sec', 'transaction_speed'],
    'Nombre_Beneficiaires_Ajoutes_Heure': ['nb_beneficiaires_heure', 'beneficiaries_added_hour', 'new_beneficiaries'],
    'Pays_Beneficiaire': ['pays', 'country', 'Country', 'beneficiary_country', 'destination_country'],
    'Compte_Risque': ['compte_risque', 'risk_account', 'account_risk', 'is_risky_account'],
    'IBAN_Liste_Noire': ['iban_blacklist', 'blacklisted_iban', 'is_blacklisted'],
    'Beneficiaire_Ajoute_Recemment': ['beneficiaire_recent', 'recent_beneficiary', 'newly_added'],
    'Flag_Fraude_Potentiel': ['fraude', 'fraud', 'is_fraud', 'fraud_flag', 'label'],
    'Heure_Normale': ['heure_normale', 'normal_hour', 'is_normal_time'],
    'Beneficiaire_Creation_Date': ['beneficiaire_creation', 'beneficiary_created', 'creation_date']
}

# Colonnes obligatoires minimum
COLONNES_OBLIGATOIRES = ['Client_ID', 'Date_Transaction', 'Montant']

# ------------------------------------------------------------------
# SCORES PAR USE CASE (chargés depuis config_fraude.yaml ou valeurs par défaut)
# ------------------------------------------------------------------
scores_config = CONFIG.get('scores', {})
SCORE_MONTANT_INHABITUEL = scores_config.get('montant_inhabituel', 30)
SCORE_HORAIRE_INHABITUEL = scores_config.get('horaire_inhabituel', 15)
SCORE_LOCALISATION_INCOHERENTE = scores_config.get('localisation_incoherente', 30)
SCORE_VITESSE_ANORMALE = scores_config.get('vitesse_anormale', 20)
SCORE_PAYS_A_RISQUE = scores_config.get('pays_a_risque', 25)
SCORE_AJOUT_MASSIF_BENEFICIAIRES = scores_config.get('ajout_massif_beneficiaires', 20)
SCORE_TRANSACTION_IMMEDIATE_APRES_AJOUT = scores_config.get('transaction_immediate_apres_ajout', 30)
SCORE_BENEFICIAIRE_BLACKLISTE = scores_config.get('beneficiaire_blackliste', 50)

# ------------------------------------------------------------------
# SEUILS DE RISQUE FINAL (chargés depuis config_fraude.yaml ou valeurs par défaut)
# ------------------------------------------------------------------
risque_config = CONFIG.get('risque', {})
RISQUE_FAIBLE_MAX = risque_config.get('faible_max', 39)
RISQUE_MOYEN_MAX = risque_config.get('moyen_max', 69)


# ==================================================================
# 1. CHARGEMENT ET PREPARATION
# ==================================================================
def normaliser_noms_colonnes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise les noms de colonnes en utilisant le mapping flexible.
    Retourne le DataFrame avec les noms de colonnes standardisés.
    """
    logger.info(f"Colonnes originales: {list(df.columns)}")
    
    # Créer un dictionnaire de renommage
    rename_dict = {}
    colonnes_trouvees = set()
    
    for colonne_standard, alternatives in COLONNE_MAPPING.items():
        for alt in alternatives:
            if alt in df.columns and colonne_standard not in df.columns:
                rename_dict[alt] = colonne_standard
                colonnes_trouvees.add(colonne_standard)
                logger.info(f"Mapping: '{alt}' -> '{colonne_standard}'")
                break
        # Si la colonne standard existe déjà, on la garde
        if colonne_standard in df.columns:
            colonnes_trouvees.add(colonne_standard)
    
    # Renommer les colonnes
    if rename_dict:
        df = df.rename(columns=rename_dict)
    
    logger.info(f"Colonnes après normalisation: {list(df.columns)}")
    logger.info(f"Colonnes trouvées: {colonnes_trouvees}")
    
    return df, colonnes_trouvees


def valider_et_completer_donnees(df: pd.DataFrame, colonnes_trouvees: set) -> pd.DataFrame:
    """
    Valide les données obligatoires et complète les colonnes manquantes avec des valeurs par défaut.
    """
    # Vérifier les colonnes obligatoires
    colonnes_manquantes = [col for col in COLONNES_OBLIGATOIRES if col not in colonnes_trouvees]
    if colonnes_manquantes:
        raise ValueError(f"Colonnes obligatoires manquantes: {colonnes_manquantes}. "
                        f"Colonnes disponibles: {list(df.columns)}")
    
    # Compléter les colonnes optionnelles manquantes avec des valeurs par défaut
    valeurs_par_defaut = {
        'Heure_Transaction': '00:00:00',
        'Distance_Derniere_Transaction_KM': 0,
        'Vitesse_Transactions_Secondes': 0,
        'Nombre_Beneficiaires_Ajoutes_Heure': 0,
        'Pays_Beneficiaire': 'Inconnu',
        'Compte_Risque': False,
        'IBAN_Liste_Noire': False,
        'Beneficiaire_Ajoute_Recemment': False,
        'Flag_Fraude_Potentiel': False,
        'Heure_Normale': True,
        'Beneficiaire_Creation_Date': None
    }
    
    for colonne, valeur_defaut in valeurs_par_defaut.items():
        if colonne not in colonnes_trouvees:
            df[colonne] = valeur_defaut
            logger.warning(f"Colonne '{colonne}' manquante, valeur par défaut: {valeur_defaut}")
    
    return df


def charger_donnees(path: str) -> pd.DataFrame:
    """
    Charge les données depuis un fichier CSV avec gestion flexible des colonnes.
    Gère différents formats de données et valeurs manquantes.
    """
    try:
        df = pd.read_csv(path)
        logger.info(f"Chargement réussi: {len(df)} lignes, {len(df.columns)} colonnes")
    except FileNotFoundError:
        logger.error(f"Fichier non trouvé: {path}")
        raise
    except Exception as e:
        logger.error(f"Erreur lors du chargement: {e}")
        raise
    
    # Normaliser les noms de colonnes
    df, colonnes_trouvees = normaliser_noms_colonnes(df)
    
    # Valider et compléter les données
    df = valider_et_completer_donnees(df, colonnes_trouvees)

    # Normalisation des types booléens
    bool_cols = ["Heure_Normale", "Compte_Risque", "IBAN_Liste_Noire",
                 "Beneficiaire_Ajoute_Recemment", "Flag_Fraude_Potentiel"]
    for c in bool_cols:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip().str.lower().map(
                {"true": True, "false": False, "1": True, "0": False, "yes": True, "no": False}
            ).fillna(False)

    # Normalisation des dates avec gestion des erreurs
    df["Date_Transaction"] = pd.to_datetime(df["Date_Transaction"], errors="coerce")
    
    # Gestion flexible de l'heure (peut être dans Date_Transaction ou séparée)
    if "Heure_Transaction" in df.columns:
        df["Heure_Transaction"] = pd.to_datetime(df["Heure_Transaction"], format="%H:%M:%S", errors="coerce").dt.time
    else:
        # Extraire l'heure de Date_Transaction si disponible
        df["Heure_Transaction"] = pd.to_datetime(df["Date_Transaction"], errors="coerce").dt.time
    
    # Nettoyage des valeurs manquantes dans les colonnes numériques
    colonnes_numeriques = ['Montant', 'Distance_Derniere_Transaction_KM', 
                          'Vitesse_Transactions_Secondes', 'Nombre_Beneficiaires_Ajoutes_Heure']
    for col in colonnes_numeriques:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Nettoyage des valeurs manquantes dans les colonnes catégorielles
    if 'Pays_Beneficiaire' in df.columns:
        df['Pays_Beneficiaire'] = df['Pays_Beneficiaire'].fillna('Inconnu')

    df = df.sort_values(["Client_ID", "Date_Transaction", "Heure_Transaction"]).reset_index(drop=True)
    
    logger.info(f"Données préparées: {len(df)} transactions, {df['Client_ID'].nunique()} clients uniques")
    
    return df


def charger_donnees_json(transaction_dict: Dict) -> pd.DataFrame:
    """
    Charge les données depuis un dictionnaire JSON pour une seule transaction.
    """
    df = pd.DataFrame([transaction_dict])
    logger.info(f"Chargement réussi depuis JSON: 1 transaction")
    
    # Normaliser les noms de colonnes
    df, colonnes_trouvees = normaliser_noms_colonnes(df)
    
    # Valider et compléter les données
    df = valider_et_completer_donnees(df, colonnes_trouvees)

    # Normalisation des types booléens
    bool_cols = ["Heure_Normale", "Compte_Risque", "IBAN_Liste_Noire",
                 "Beneficiaire_Ajoute_Recemment", "Flag_Fraude_Potentiel"]
    for c in bool_cols:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip().str.lower().map(
                {"true": True, "false": False, "1": True, "0": False, "yes": True, "no": False}
            ).fillna(False)

    # Normalisation des dates avec gestion des erreurs
    df["Date_Transaction"] = pd.to_datetime(df["Date_Transaction"], errors="coerce")
    
    # Gestion flexible de l'heure
    if "Heure_Transaction" in df.columns:
        df["Heure_Transaction"] = pd.to_datetime(df["Heure_Transaction"], format="%H:%M:%S", errors="coerce").dt.time
    else:
        df["Heure_Transaction"] = pd.to_datetime(df["Date_Transaction"], errors="coerce").dt.time
    
    # Normalisation des dates de création de bénéficiaire
    if "Beneficiaire_Creation_Date" in df.columns:
        df["Beneficiaire_Creation_Date"] = pd.to_datetime(df["Beneficiaire_Creation_Date"], errors="coerce")
    
    # Nettoyage des valeurs manquantes dans les colonnes numériques
    colonnes_numeriques = ['Montant', 'Distance_Derniere_Transaction_KM', 
                          'Vitesse_Transactions_Secondes', 'Nombre_Beneficiaires_Ajoutes_Heure']
    for col in colonnes_numeriques:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Nettoyage des valeurs manquantes dans les colonnes catégorielles
    if 'Pays_Beneficiaire' in df.columns:
        df['Pays_Beneficiaire'] = df['Pays_Beneficiaire'].fillna('Inconnu')

    df = df.sort_values(["Client_ID", "Date_Transaction", "Heure_Transaction"]).reset_index(drop=True)
    
    logger.info(f"Données préparées: 1 transaction")
    
    return df


def charger_fichier_json(path_json: str) -> Union[Dict, List[Dict]]:
    """
    Charge un fichier JSON contenant une seule transaction ou une liste de transactions.
    """
    with open(path_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    logger.info(f"Fichier JSON chargé depuis: {path_json}")
    return data


# ==================================================================
# 2. REGLES METIER (une fonction = un cas d'usage demande)
# ==================================================================

def regle_montant_inhabituel(df):
    """
    Cas 1 : Montant inhabituel par rapport a l'historique du client (z-score).
    IMPORTANT : on compare chaque transaction a l'historique QUI LA PRECEDE
    (fenetre "expanding" decalee), jamais a une moyenne qui inclut la
    transaction elle-meme -- sinon une fraude de gros montant fait exploser
    sa propre moyenne/ecart-type et se "cache" statistiquement.
    """
    if "Montant" not in df.columns:
        logger.warning("Colonne 'Montant' manquante, règle montant inhabituel désactivée")
        df["montant_moyen_historique"] = 0
        df["montant_median_historique"] = 0
        df["montant_std_historique"] = 0
        df["nb_transac_historique"] = 0
        df["zscore_montant"] = 0.0
        df["Anomalie_Montant"] = False
        return df
    
    df = df.sort_values(["Client_ID", "Date_Transaction", "Heure_Transaction"]).reset_index(drop=True)

    moyennes, medianes, ecarts, effectifs, zscores = [], [], [], [], []
    for client_id, groupe in df.groupby("Client_ID", sort=False):
        montants = groupe["Montant"].tolist()
        for i in range(len(montants)):
            historique = montants[:i]  # tout ce qui precede, EXCLUT la transaction courante
            n = len(historique)
            if n < 3:
                moyennes.append(np.mean(historique) if n > 0 else 0)
                medianes.append(np.median(historique) if n > 0 else 0)
                ecarts.append(0)
                effectifs.append(n)
                zscores.append(0.0)
                continue
            m = np.mean(historique)
            med = np.median(historique)
            s = np.std(historique, ddof=1)
            moyennes.append(m)
            medianes.append(med)
            ecarts.append(s)
            effectifs.append(n)
            zscores.append(abs(montants[i] - m) / s if s > 0 else (5.0 if montants[i] != m else 0.0))

    df["montant_moyen_historique"] = moyennes
    df["montant_median_historique"] = medianes
    df["montant_std_historique"] = ecarts
    df["nb_transac_historique"] = effectifs
    df["zscore_montant"] = zscores
    df["Anomalie_Montant"] = df["zscore_montant"] > SEUIL_Z_SCORE_MONTANT
    return df


def regle_horaire_inhabituel(df):
    """
    Cas 2 : Transaction a une heure inhabituelle.
    Construit le profil horaire du client et detecte les heures rares (<5% des habitudes).
    """
    if "Heure_Transaction" not in df.columns:
        logger.warning("Colonne 'Heure_Transaction' manquante, règle horaire inhabituel désactivée")
        df["Anomalie_Horaire"] = False
        return df
    
    df = df.sort_values(["Client_ID", "Date_Transaction", "Heure_Transaction"]).reset_index(drop=True)
    
    heures_rare = []
    for client_id, groupe in df.groupby("Client_ID", sort=False):
        heures = [h.hour if pd.notna(h) else 0 for h in groupe["Heure_Transaction"]]
        for i in range(len(heures)):
            historique = heures[:i]  # tout ce qui precede
            n = len(historique)
            if n < 10:
                heures_rare.append(False)
                continue
            # Calculer la frequence de chaque heure dans l'historique
            from collections import Counter
            freq = Counter(historique)
            heure_courante = heures[i]
            # Si cette heure represente moins de SEUIL_HEURE_RARE_PCT des transactions
            if freq.get(heure_courante, 0) / n < SEUIL_HEURE_RARE_PCT:
                heures_rare.append(True)
            else:
                heures_rare.append(False)
    
    df["Anomalie_Horaire"] = heures_rare
    return df


def regle_localisation_incoherente(df):
    """
    Cas 3 : Géolocalisation incohérente.
    Calcule la vitesse entre deux localisations successives.
    Vitesse = Distance / Temps. Si vitesse impossible -> alerte.
    """
    if "Distance_Derniere_Transaction_KM" not in df.columns:
        logger.warning("Colonne 'Distance_Derniere_Transaction_KM' manquante, règle localisation incohérente désactivée")
        df["Vitesse_KMH"] = 0
        df["Anomalie_Localisation"] = False
        return df
    
    df = df.sort_values(["Client_ID", "Date_Transaction", "Heure_Transaction"]).reset_index(drop=True)
    
    vitesses = []
    localisation_incoherente = []
    
    for client_id, groupe in df.groupby("Client_ID", sort=False):
        for i in range(len(groupe)):
            if i == 0:
                vitesses.append(0)
                localisation_incoherente.append(False)
                continue
            
            # Recuperer infos transaction courante et precedente
            row_curr = groupe.iloc[i]
            row_prev = groupe.iloc[i-1]
            
            distance = row_curr.get("Distance_Derniere_Transaction_KM", 0)
            if pd.isna(distance):
                distance = 0
            
            # Calculer le temps ecoule en secondes
            dt_curr = pd.to_datetime(row_curr["Date_Transaction"])
            dt_prev = pd.to_datetime(row_prev["Date_Transaction"])
            
            # Ajouter l'heure si disponible
            if pd.notna(row_curr.get("Heure_Transaction")):
                dt_curr = pd.to_datetime(str(dt_curr.date()) + " " + str(row_curr["Heure_Transaction"]))
            if pd.notna(row_prev.get("Heure_Transaction")):
                dt_prev = pd.to_datetime(str(dt_prev.date()) + " " + str(row_prev["Heure_Transaction"]))
            
            temps_ecoule = (dt_curr - dt_prev).total_seconds()
            
            if temps_ecoule <= 0:
                vitesses.append(0)
                localisation_incoherente.append(False)
                continue
            
            # Vitesse en km/h = (km / secondes) * 3600
            vitesse_kmh = (distance / temps_ecoule) * 3600 if temps_ecoule > 0 else 0
            vitesses.append(vitesse_kmh)
            
            # Si vitesse impossible
            localisation_incoherente.append(vitesse_kmh > SEUIL_VITESSE_KMH_IMPOSSIBLE)
    
    df["Vitesse_KMH"] = vitesses
    df["Anomalie_Localisation"] = localisation_incoherente
    return df


def regle_vitesse_anormale(df):
    """
    Cas 4 : Vitesse anormale des transactions.
    Detecte plusieurs transactions effectuees en tres peu de temps.
    Exemple: 5 transactions en 2 minutes ou 10 transactions en 5 minutes.
    """
    if "Date_Transaction" not in df.columns:
        logger.warning("Colonne 'Date_Transaction' manquante, règle vitesse anormale désactivée")
        df["Anomalie_Vitesse"] = False
        return df
    
    df = df.sort_values(["Client_ID", "Date_Transaction", "Heure_Transaction"]).reset_index(drop=True)
    
    vitesse_anormale = []
    
    for client_id, groupe in df.groupby("Client_ID", sort=False):
        for i in range(len(groupe)):
            # Compter le nombre de transactions dans la fenetre temporelle precedente
            dt_curr = pd.to_datetime(groupe.iloc[i]["Date_Transaction"])
            if pd.notna(groupe.iloc[i].get("Heure_Transaction")):
                dt_curr = pd.to_datetime(str(dt_curr.date()) + " " + str(groupe.iloc[i]["Heure_Transaction"]))
            
            dt_debut = dt_curr - pd.Timedelta(seconds=SEUIL_FENETRE_TEMPS)
            
            # Compter transactions dans cette fenetre (excluant la courante)
            nb_trans_fenetre = 0
            for j in range(i):
                dt_j = pd.to_datetime(groupe.iloc[j]["Date_Transaction"])
                if pd.notna(groupe.iloc[j].get("Heure_Transaction")):
                    dt_j = pd.to_datetime(str(dt_j.date()) + " " + str(groupe.iloc[j]["Heure_Transaction"]))
                if dt_debut <= dt_j <= dt_curr:
                    nb_trans_fenetre += 1
            
            vitesse_anormale.append(nb_trans_fenetre >= SEUIL_NB_TRANS_FENETRE)
    
    df["Anomalie_Vitesse"] = vitesse_anormale
    return df


def regle_pays_a_risque(df):
    """
    Cas 5 : Transaction vers un pays a risque.
    Compare le pays du beneficiaire avec une liste de pays a risque.
    """
    if "Pays_Beneficiaire" not in df.columns:
        logger.warning("Colonne 'Pays_Beneficiaire' manquante, règle pays à risque basée sur Compte_Risque uniquement")
        df["Anomalie_Pays_Risque"] = df["Compte_Risque"].astype(bool) if "Compte_Risque" in df.columns else False
        return df
    
    pays_susp = df["Pays_Beneficiaire"].isin(PAYS_A_RISQUE)
    compte_risque = df["Compte_Risque"].astype(bool) if "Compte_Risque" in df.columns else False
    df["Anomalie_Pays_Risque"] = compte_risque | pays_susp
    return df


def regle_ajout_massif_beneficiaires(df):
    """
    Cas 6 : Ajout massif de beneficiaires.
    Detecte un nombre anormal de beneficiaires ajoutes en peu de temps.
    Exemple: Plus de 5 beneficiaires ajoutes en 30 minutes.
    """
    if "Nombre_Beneficiaires_Ajoutes_Heure" not in df.columns:
        logger.warning("Colonne 'Nombre_Beneficiaires_Ajoutes_Heure' manquante, règle ajout massif désactivée")
        df["Anomalie_Ajout_Massif_Beneficiaires"] = False
        return df
    
    df["Anomalie_Ajout_Massif_Beneficiaires"] = df["Nombre_Beneficiaires_Ajoutes_Heure"].fillna(0) >= SEUIL_NB_BENEFICIAIRES_HEURE
    return df


def regle_transaction_immediate_apres_ajout(df):
    """
    Cas 7 : Transaction immediate apres ajout de beneficiaire.
    Detecte une transaction effectuee juste apres l'ajout d'un beneficiaire.
    Exemple: Ajout a 14h00, transaction a 14h03 (3 minutes) -> alerte si < 10 min.
    """
    if "Beneficiaire_Ajoute_Recemment" not in df.columns:
        logger.warning("Colonne 'Beneficiaire_Ajoute_Recemment' manquante, règle transaction immédiate désactivée")
        df["Anomalie_Transaction_Immediate_Apres_Ajout"] = False
        return df
    
    df = df.sort_values(["Client_ID", "Date_Transaction", "Heure_Transaction"]).reset_index(drop=True)
    
    transaction_immediate = []
    
    for client_id, groupe in df.groupby("Client_ID", sort=False):
        for i in range(len(groupe)):
            row = groupe.iloc[i]
            if not row.get("Beneficiaire_Ajoute_Recemment", False):
                transaction_immediate.append(False)
                continue
            
            # Calculer le delai entre l'ajout et la transaction
            dt_transaction = pd.to_datetime(row["Date_Transaction"])
            if pd.notna(row.get("Heure_Transaction")):
                dt_transaction = pd.to_datetime(str(dt_transaction.date()) + " " + str(row["Heure_Transaction"]))
            
            # Si on a une date de creation du beneficiaire
            if "Beneficiaire_Creation_Date" in df.columns and pd.notna(row.get("Beneficiaire_Creation_Date")):
                dt_creation = pd.to_datetime(row["Beneficiaire_Creation_Date"])
                delai_secondes = (dt_transaction - dt_creation).total_seconds()
                transaction_immediate.append(delai_secondes < SEUIL_DELAI_AJOUT_TRANSACTION)
            else:
                # Fallback: utiliser la vitesse entre transactions comme approximation
                vitesse = row.get("Vitesse_Transactions_Secondes", 1e9)
                if pd.isna(vitesse):
                    vitesse = 1e9
                transaction_immediate.append(vitesse < SEUIL_DELAI_AJOUT_TRANSACTION)
    
    df["Anomalie_Transaction_Immediate_Apres_Ajout"] = transaction_immediate
    return df


def regle_iban_blackliste(df):
    """Cas 7 : Beneficiaire avec IBAN/compte deja signale en liste noire."""
    if "IBAN_Liste_Noire" not in df.columns:
        logger.warning("Colonne 'IBAN_Liste_Noire' manquante, règle IBAN blacklist désactivée")
        df["Anomalie_IBAN_Blackliste"] = False
        return df
    
    df["Anomalie_IBAN_Blackliste"] = df["IBAN_Liste_Noire"].astype(bool)
    return df


def appliquer_toutes_les_regles(df):
    df = regle_montant_inhabituel(df)
    df = regle_horaire_inhabituel(df)
    df = regle_localisation_incoherente(df)
    df = regle_vitesse_anormale(df)
    df = regle_pays_a_risque(df)
    df = regle_ajout_massif_beneficiaires(df)
    df = regle_transaction_immediate_apres_ajout(df)
    df = regle_iban_blackliste(df)
    return df


# ==================================================================
# 3. MACHINE LEARNING
# ==================================================================
FEATURES_ML = [
    "Montant", "Distance_Derniere_Transaction_KM", "Vitesse_Transactions_Secondes",
    "Nombre_Beneficiaires_Ajoutes_Heure", "Score_Anomalie_Montant",
    "Score_Anomalie_Temps", "Score_Anomalie_Lieu", "Score_Risque_Global",
    "zscore_montant"
]


def construire_matrice_features(df):
    """
    Construit la matrice de features pour le ML avec gestion des colonnes manquantes.
    Utilise uniquement les colonnes disponibles dans le DataFrame.
    """
    X = df.copy()
    
    # Colonnes booléennes à convertir en numériques
    bool_cols = ["Compte_Risque", "IBAN_Liste_Noire", "Beneficiaire_Ajoute_Recemment", "Heure_Normale"]
    for col in bool_cols:
        if col in X.columns:
            X[col + "_num"] = X[col].astype(int)
        else:
            logger.warning(f"Colonne '{col}' manquante, utilisation de 0 par défaut")
            X[col + "_num"] = 0

    # Sélectionner uniquement les colonnes disponibles
    cols_disponibles = [c for c in FEATURES_ML if c in X.columns]
    cols_bool_num = [c for c in X.columns if c.endswith("_num")]
    
    cols = cols_disponibles + cols_bool_num
    
    if not cols:
        logger.error("Aucune feature disponible pour le ML")
        raise ValueError("Aucune colonne utilisable pour l'entraînement ML")
    
    X = X[cols].fillna(0)
    logger.info(f"Features ML utilisées: {cols}")
    
    return X


def modele_isolation_forest(df):
    """Detection non supervisee : apprend le comportement normal, signale les ecarts."""
    X = construire_matrice_features(df)
    if len(X) < 10:
        # Pas assez de donnees pour un vrai entrainement -> on le signale clairement
        df["Score_ML_IsolationForest"] = 0.0
        df["Anomalie_ML_NonSupervisee"] = False
        df["_ml_warning"] = "Dataset trop petit (<10 lignes) pour entrainer un Isolation Forest fiable"
        return df

    # Charger les paramètres depuis la configuration
    iso_config = ml_config.get('isolation_forest', {})
    n_estimators = iso_config.get('n_estimators', 200)
    random_state = iso_config.get('random_state', 42)

    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=CONTAMINATION_ISOLATION_FOREST,
        random_state=random_state
    )
    model.fit(X)
    # decision_function : plus c'est negatif, plus c'est anormal -> on inverse pour avoir un score d'anomalie positif
    raw_scores = -model.decision_function(X)
    df["Score_ML_IsolationForest"] = (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min() + 1e-9)
    df["Anomalie_ML_NonSupervisee"] = model.predict(X) == -1
    df["_ml_warning"] = ""
    return df


def modele_random_forest_supervise(df):
    """
    Detection supervisee : si vous avez un historique de fraudes confirmees
    (colonne Flag_Fraude_Potentiel avec assez d'exemples des 2 classes),
    on entraine un Random Forest pour apprendre les patterns de fraude reels.
    """
    df["Score_ML_Supervise"] = np.nan
    df["_rf_warning"] = ""

    if "Flag_Fraude_Potentiel" not in df.columns:
        df["_rf_warning"] = "Colonne Flag_Fraude_Potentiel absente : modele supervise non entraine"
        return df, None

    y = df["Flag_Fraude_Potentiel"].astype(int)
    if y.nunique() < 2 or len(df) < 30:
        df["_rf_warning"] = ("Pas assez de donnees/diversite de labels pour entrainer un Random Forest fiable "
                              "(il faut idealement plusieurs centaines de transactions avec des cas fraude ET non-fraude). "
                              "Le score supervise n'est donc pas calcule sur cet echantillon de demonstration.")
        return df, None

    X = construire_matrice_features(df)
    
    # Charger les paramètres depuis la configuration
    rf_config = ml_config.get('random_forest', {})
    n_estimators = rf_config.get('n_estimators', 300)
    max_depth = rf_config.get('max_depth', 8)
    class_weight = rf_config.get('class_weight', 'balanced')
    random_state = rf_config.get('random_state', 42)
    
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight=class_weight,   # important : la fraude est toujours minoritaire
        random_state=random_state
    )
    model.fit(X, y)
    df["Score_ML_Supervise"] = model.predict_proba(X)[:, 1]
    return df, model


# ==================================================================
# 4. FUSION DES SIGNAUX -> SCORE DE RISQUE
# ==================================================================
COLONNES_REGLES = [
    "Anomalie_Montant", "Anomalie_Horaire", "Anomalie_Localisation", "Anomalie_Vitesse",
    "Anomalie_Pays_Risque", "Anomalie_Ajout_Massif_Beneficiaires",
    "Anomalie_Transaction_Immediate_Apres_Ajout", "Anomalie_IBAN_Blackliste"
]

# Mapping des scores par use case
SCORES_PAR_REGLE = {
    "Anomalie_Montant": SCORE_MONTANT_INHABITUEL,
    "Anomalie_Horaire": SCORE_HORAIRE_INHABITUEL,
    "Anomalie_Localisation": SCORE_LOCALISATION_INCOHERENTE,
    "Anomalie_Vitesse": SCORE_VITESSE_ANORMALE,
    "Anomalie_Pays_Risque": SCORE_PAYS_A_RISQUE,
    "Anomalie_Ajout_Massif_Beneficiaires": SCORE_AJOUT_MASSIF_BENEFICIAIRES,
    "Anomalie_Transaction_Immediate_Apres_Ajout": SCORE_TRANSACTION_IMMEDIATE_APRES_AJOUT,
    "Anomalie_IBAN_Blackliste": SCORE_BENEFICIAIRE_BLACKLISTE
}


def calculer_score_risque(df):
    """
    Calcule le score de risque total en sommant les scores de chaque use case declenche.
    Classification du risque:
    - 0-39 : risque faible (pas d'alerte)
    - 40-69 : risque moyen (alerte simple ou verification)
    - >= 70 : risque eleve (blocage ou validation forte)
    """
    df["Score_Risque"] = 0
    
    for regle, score in SCORES_PAR_REGLE.items():
        if regle in df.columns:
            df["Score_Risque"] += df[regle].astype(int) * score
    
    # Classification du risque
    def classer_risque(score):
        if score <= RISQUE_FAIBLE_MAX:
            return "FAIBLE"
        elif score <= RISQUE_MOYEN_MAX:
            return "MOYEN"
        else:
            return "ELEVE"
    
    df["Niveau_Risque"] = df["Score_Risque"].apply(classer_risque)
    df["Alerte"] = df["Score_Risque"] > RISQUE_FAIBLE_MAX
    
    return df


def calculer_verdict_final(df):
    df["Nb_Regles_Declenchees"] = df[COLONNES_REGLES].sum(axis=1)
    
    # Calculer le score de risque
    df = calculer_score_risque(df)

    def motif(row):
        motifs = [col.replace("Anomalie_", "") for col in COLONNES_REGLES if row.get(col, False)]
        return ", ".join(motifs) if motifs else "RAS"

    df["Motif_Detecte"] = df.apply(motif, axis=1)
    df["Verdict_Texte"] = df["Niveau_Risque"]
    return df


def predire_transaction(transaction_dict: Dict, modele_isolation_forest_trained=None, modele_rf_trained=None) -> Dict:
    """
    Prédit le risque de fraude pour une seule transaction donnée sous forme de dictionnaire JSON.
    
    Args:
        transaction_dict: Dictionnaire contenant les données de la transaction
        modele_isolation_forest_trained: Modèle Isolation Forest pré-entraîné (optionnel)
        modele_rf_trained: Modèle Random Forest pré-entraîné (optionnel)
    
    Returns:
        Dictionnaire contenant le résultat de la prédiction
    """
    # Charger les données depuis le dictionnaire
    df = charger_donnees_json(transaction_dict)
    
    # Appliquer les règles métier
    df = appliquer_toutes_les_regles(df)
    
    # Utiliser les modèles pré-entraînés si fournis, sinon entraîner sur la donnée unique
    if modele_isolation_forest_trained is not None:
        X = construire_matrice_features(df)
        raw_scores = -modele_isolation_forest_trained.decision_function(X)
        df["Score_ML_IsolationForest"] = (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min() + 1e-9)
        df["Anomalie_ML_NonSupervisee"] = modele_isolation_forest_trained.predict(X) == -1
        df["_ml_warning"] = ""
    else:
        df = modele_isolation_forest(df)
    
    if modele_rf_trained is not None:
        X = construire_matrice_features(df)
        df["Score_ML_Supervise"] = modele_rf_trained.predict_proba(X)[:, 1]
        df["_rf_warning"] = ""
    else:
        df, _ = modele_random_forest_supervise(df)
    
    # Calculer le verdict final
    df = calculer_verdict_final(df)
    
    # Extraire le résultat pour la transaction unique
    resultat = {
        "Client_ID": df.iloc[0]["Client_ID"],
        "Date_Transaction": str(df.iloc[0]["Date_Transaction"]),
        "Heure_Transaction": str(df.iloc[0]["Heure_Transaction"]),
        "Montant": float(df.iloc[0]["Montant"]),
        "Score_Risque": int(df.iloc[0]["Score_Risque"]),
        "Niveau_Risque": df.iloc[0]["Niveau_Risque"],
        "Alerte": bool(df.iloc[0]["Alerte"]),
        "Motif_Detecte": df.iloc[0]["Motif_Detecte"],
        "Verdict_Texte": df.iloc[0]["Verdict_Texte"],
        "Nb_Regles_Declenchees": int(df.iloc[0]["Nb_Regles_Declenchees"]),
        "Details_Regles": {}
    }
    
    # Ajouter les détails de chaque règle
    for regle in COLONNES_REGLES:
        if regle in df.columns:
            resultat["Details_Regles"][regle] = bool(df.iloc[0][regle])
    
    # Ajouter les scores ML si disponibles
    if "Score_ML_IsolationForest" in df.columns:
        resultat["Score_ML_IsolationForest"] = float(df.iloc[0]["Score_ML_IsolationForest"])
        resultat["Anomalie_ML_NonSupervisee"] = bool(df.iloc[0]["Anomalie_ML_NonSupervisee"])
    
    if "Score_ML_Supervise" in df.columns and pd.notna(df.iloc[0]["Score_ML_Supervise"]):
        resultat["Score_ML_Supervise"] = float(df.iloc[0]["Score_ML_Supervise"])
    
    return resultat


# ==================================================================
# 5. PIPELINE COMPLET
# ==================================================================
def executer_pipeline(path_csv, path_sortie="resultats_detection_fraude.csv"):
    print(f"[1/5] Chargement de {path_csv} ...")
    df = charger_donnees(path_csv)
    print(f"      -> {len(df)} transactions chargees, {df['Client_ID'].nunique()} clients uniques")

    print("[2/5] Application des regles metier (7 cas d'usage) ...")
    df = appliquer_toutes_les_regles(df)

    print("[3/5] Entrainement Isolation Forest (non supervise) ...")
    df = modele_isolation_forest(df)

    print("[4/5] Entrainement Random Forest (supervise, si labels dispo) ...")
    df, rf_model = modele_random_forest_supervise(df)

    print("[5/5] Fusion des signaux -> verdict final ...")
    df = calculer_verdict_final(df)

    # -------- Rapport console --------
    print("\n" + "=" * 70)
    print("RESUME PAR CAS D'USAGE")
    print("=" * 70)
    noms_lisibles = {
        "Anomalie_Montant": "Montant inhabituel",
        "Anomalie_Horaire": "Horaire inhabituel",
        "Anomalie_Localisation": "Localisation incoherente",
        "Anomalie_Vitesse": "Vitesse de transaction anormale",
        "Anomalie_Pays_Risque": "Destination a risque",
        "Anomalie_Ajout_Massif_Beneficiaires": "Ajout massif de beneficiaires",
        "Anomalie_Transaction_Immediate_Apres_Ajout": "Transaction immediate apres ajout",
        "Anomalie_IBAN_Blackliste": "Beneficiaire blackliste",
    }
    for col, label in noms_lisibles.items():
        if col in df.columns:
            nb = int(df[col].sum())
            score = SCORES_PAR_REGLE.get(col, 0)
            print(f"  - {label:45s} : {nb} transaction(s) suspecte(s) (score: +{score})")

    print("-" * 70)
    print(f"  DISTRIBUTION DU RISQUE:")
    print(f"    - FAIBLE (0-{RISQUE_FAIBLE_MAX})       : {(df['Niveau_Risque'] == 'FAIBLE').sum()} transactions")
    print(f"    - MOYEN ({RISQUE_FAIBLE_MAX+1}-{RISQUE_MOYEN_MAX})     : {(df['Niveau_Risque'] == 'MOYEN').sum()} transactions")
    print(f"    - ELEVE (>={RISQUE_MOYEN_MAX+1})        : {(df['Niveau_Risque'] == 'ELEVE').sum()} transactions")
    print(f"  ALERTES DECLENCHEES : {int(df['Alerte'].sum())} / {len(df)} ({df['Alerte'].mean()*100:.1f}%)")
    print("=" * 70)

    if df["_ml_warning"].iloc[0] if "_ml_warning" in df.columns else None:
        if df["_ml_warning"].iloc[0]:
            print(f"\n[Note ML non supervise] {df['_ml_warning'].iloc[0]}")
    if "_rf_warning" in df.columns and df["_rf_warning"].iloc[0]:
        print(f"[Note ML supervise]     {df['_rf_warning'].iloc[0]}")

    # -------- Detail ligne par ligne --------
    print("\nDETAIL PAR TRANSACTION :")
    colonnes_affichage = ["Client_ID", "Date_Transaction", "Montant", "Score_Risque", "Niveau_Risque", "Motif_Detecte"]
    print(df[colonnes_affichage].to_string(index=False))

    # -------- Sauvegarde --------
    colonnes_sortie = [
        "Client_ID", "Date_Transaction", "Heure_Transaction", "Montant",
    ] + COLONNES_REGLES + [
        "Score_Risque", "Niveau_Risque", "Alerte",
        "Score_ML_IsolationForest", "Anomalie_ML_NonSupervisee",
        "Score_ML_Supervise", "Nb_Regles_Declenchees", "Motif_Detecte", "Verdict_Texte"
    ]
    colonnes_sortie = [c for c in colonnes_sortie if c in df.columns]
    df[colonnes_sortie].to_csv(path_sortie, index=False)
    print(f"\nResultats detailles sauvegardes dans : {path_sortie}")

    return df, rf_model


if __name__ == "__main__":
    # Parsing des arguments en ligne de commande
    # Usage: python fraud_detection_model.py <fichier_donnees> [fichier_config]
    if len(sys.argv) < 2:
        print("Usage: python fraud_detection_model.py <fichier_donnees> [fichier_config]")
        print("Exemple CSV: python fraud_detection_model.py data/mon_dataset.csv")
        print("Exemple JSON: python fraud_detection_model.py transaction.json")
        sys.exit(1)
    
    chemin_donnees = sys.argv[1]
    chemin_config = sys.argv[2] if len(sys.argv) > 2 else "config_fraude.yaml"
    
    # Recharger la configuration avec le chemin spécifié
    if len(sys.argv) > 2:
        CONFIG = charger_config(chemin_config)
        logger.info(f"Configuration chargée depuis: {chemin_config}")
    
    # Détecter si c'est un fichier JSON ou CSV
    if chemin_donnees.endswith('.json'):
        # Mode JSON
        data = charger_fichier_json(chemin_donnees)
        
        # Détecter si c'est une liste de transactions ou une seule transaction
        if isinstance(data, list):
            # Mode batch JSON (liste de transactions)
            print("[INFO] Mode batch JSON détecté : plusieurs transactions")
            df = pd.DataFrame(data)
            
            # Normaliser les noms de colonnes
            df, colonnes_trouvees = normaliser_noms_colonnes(df)
            
            # Valider et compléter les données
            df = valider_et_completer_donnees(df, colonnes_trouvees)
            
            # Normalisation des types booléens
            bool_cols = ["Heure_Normale", "Compte_Risque", "IBAN_Liste_Noire",
                         "Beneficiaire_Ajoute_Recemment", "Flag_Fraude_Potentiel"]
            for c in bool_cols:
                if c in df.columns:
                    df[c] = df[c].astype(str).str.strip().str.lower().map(
                        {"true": True, "false": False, "1": True, "0": False, "yes": True, "no": False}
                    ).fillna(False)
            
            # Normalisation des dates
            df["Date_Transaction"] = pd.to_datetime(df["Date_Transaction"], errors="coerce")
            if "Heure_Transaction" in df.columns:
                df["Heure_Transaction"] = pd.to_datetime(df["Heure_Transaction"], format="%H:%M:%S", errors="coerce").dt.time
            else:
                df["Heure_Transaction"] = pd.to_datetime(df["Date_Transaction"], errors="coerce").dt.time
            
            if "Beneficiaire_Creation_Date" in df.columns:
                df["Beneficiaire_Creation_Date"] = pd.to_datetime(df["Beneficiaire_Creation_Date"], errors="coerce")
            
            # Nettoyage des valeurs manquantes
            colonnes_numeriques = ['Montant', 'Distance_Derniere_Transaction_KM', 
                                  'Vitesse_Transactions_Secondes', 'Nombre_Beneficiaires_Ajoutes_Heure']
            for col in colonnes_numeriques:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
            if 'Pays_Beneficiaire' in df.columns:
                df['Pays_Beneficiaire'] = df['Pays_Beneficiaire'].fillna('Inconnu')
            
            df = df.sort_values(["Client_ID", "Date_Transaction", "Heure_Transaction"]).reset_index(drop=True)
            
            # Appliquer le pipeline complet
            print(f"[1/5] {len(df)} transactions chargées depuis JSON")
            print("[2/5] Application des regles metier (7 cas d'usage) ...")
            df = appliquer_toutes_les_regles(df)
            
            print("[3/5] Entrainement Isolation Forest (non supervise) ...")
            df = modele_isolation_forest(df)
            
            print("[4/5] Entrainement Random Forest (supervise, si labels dispo) ...")
            df, rf_model = modele_random_forest_supervise(df)
            
            print("[5/5] Fusion des signaux -> verdict final ...")
            df = calculer_verdict_final(df)
            
            # Sauvegarder le résultat
            chemin_sortie = chemin_donnees.replace('.json', '_resultat.csv')
            colonnes_sortie = [
                "Client_ID", "Date_Transaction", "Heure_Transaction", "Montant",
            ] + COLONNES_REGLES + [
                "Score_Risque", "Niveau_Risque", "Alerte",
                "Score_ML_IsolationForest", "Anomalie_ML_NonSupervisee",
                "Score_ML_Supervise", "Nb_Regles_Declenchees", "Motif_Detecte", "Verdict_Texte"
            ]
            colonnes_sortie = [c for c in colonnes_sortie if c in df.columns]
            df[colonnes_sortie].to_csv(chemin_sortie, index=False)
            
            print(f"\nResultat sauvegarde dans : {chemin_sortie}")
            print(f"Total transactions : {len(df)}")
            print(f"Alertes declenchees : {int(df['Alerte'].sum())} ({df['Alerte'].mean()*100:.1f}%)")
            
        else:
            # Mode transaction unique JSON
            print("[INFO] Mode transaction unique JSON détecté")
            resultat = predire_transaction(data)
            
            # Sauvegarder le résultat dans un fichier CSV
            df_resultat = pd.DataFrame([resultat])
            chemin_sortie = chemin_donnees.replace('.json', '_resultat.csv')
            df_resultat.to_csv(chemin_sortie, index=False)
            
            print("\n" + "="*70)
            print("RESULTAT DE LA DETECTION DE FRAUDE")
            print("="*70)
            print(json.dumps(resultat, indent=2, ensure_ascii=False))
            print("="*70)
            print(f"\nResultat sauvegarde dans : {chemin_sortie}")
    else:
        # Mode CSV batch (existant)
        executer_pipeline(chemin_donnees, "resultats_detection_fraude.csv")
        
        # Relancement du pipeline pour entraîner le modèle Random Forest
        print("\nRelancement du pipeline pour entraîner le modèle Random Forest...")
        executer_pipeline("data/mon_dataset.csv", "resultats_detection_fraude.csv")
        
        # Charger les résultats complets
        df_resultats = pd.read_csv('resultats_detection_fraude.csv')
        
        # Filtrer pour garder ABSOLUMENT TOUS les risques MOYEN et ELEVE
        df_filtre = df_resultats[df_resultats['Niveau_Risque'].isin(['MOYEN', 'ELEVE'])][['Client_ID', 'Niveau_Risque', 'Motif_Detecte', 'Score_Risque']]
        
        # Sauvegarder le fichier complet sans restriction
        chemin_filtre = 'resultats_detection_fraude_filtres.csv'
        df_filtre.to_csv(chemin_filtre, index=False)
        
        print(f"\nFiltrage terminé : {len(df_filtre)} lignes récupérées (Risque MOYEN ou ELEVE).")
        print(f"Fichier sauvegardé : {chemin_filtre}")
        
        # Afficher TOUTES les lignes extraites pour vérification visuelle
        pd.set_option('display.max_rows', None)
        print("\nTransactions suspectes (Risque MOYEN ou ELEVE) :")
        print(df_filtre.to_string(index=False))
