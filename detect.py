import os
import csv
import argparse
from collections import Counter
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Détection d'objets avec YOLOv8")
    parser.add_argument(
        "--model", type=str, default="yolov8n.pt",
        help="Modèle YOLO à utiliser (ex: yolov8n.pt, yolov8s.pt, yolov8m.pt)"
    )
    parser.add_argument(
        "--images_dir", type=str, default="images",
        help="Dossier contenant les images à analyser"
    )
    parser.add_argument(
        "--conf", type=float, default=0.25,
        help="Seuil de confiance minimum pour conserver une détection"
    )
    parser.add_argument(
        "--image", type=str, default=None,
        help="(Optionnel) Nom d'une seule image à traiter, au lieu de tout le dossier"
    )
    return parser.parse_args()


def get_image_list(images_dir, single_image=None):
    """Retourne la liste des chemins d'images à traiter."""
    valid_extensions = (".jpg", ".jpeg", ".png", ".bmp")

    if single_image:
        chemin = os.path.join(images_dir, single_image)
        if not os.path.exists(chemin):
            print(f"Erreur : l'image '{single_image}' n'existe pas dans '{images_dir}'.")
            return []
        return [chemin]

    if not os.path.exists(images_dir):
        print(f"Erreur : le dossier '{images_dir}' n'existe pas.")
        return []

    fichiers = [
        os.path.join(images_dir, f)
        for f in sorted(os.listdir(images_dir))
        if f.lower().endswith(valid_extensions)
    ]

    if not fichiers:
        print(f"Aucune image trouvée dans '{images_dir}'.")

    return fichiers


def main():
    args = parse_args()

    print(f"Chargement du modèle {args.model}...")
    model = YOLO(args.model)

    image_paths = get_image_list(args.images_dir, args.image)
    if not image_paths:
        return

    print(f"{len(image_paths)} image(s) à traiter avec un seuil de confiance de {args.conf}.\n")

    # Compteurs globaux pour le résumé final
    total_objets = 0
    compteur_classes = Counter()
    resume_par_image = []

    for chemin_image in image_paths:
        nom_image = os.path.basename(chemin_image)
        print(f"===== {nom_image} =====")

        results = model.predict(
            source=chemin_image,
            save=True,
            conf=args.conf,
            verbose=False
        )

        for result in results:
            if len(result.boxes) == 0:
                print("Aucun objet détecté.")
                resume_par_image.append({"image": nom_image, "classe": "aucune", "confiance": ""})
                continue

            for box in result.boxes:
                classe_id = int(box.cls[0])
                confiance = float(box.conf[0])
                nom_classe = model.names[classe_id]

                print(f"Objet détecté : {nom_classe} | Confiance : {confiance:.2%}")

                total_objets += 1
                compteur_classes[nom_classe] += 1
                resume_par_image.append({
                    "image": nom_image,
                    "classe": nom_classe,
                    "confiance": f"{confiance:.4f}"
                })

        print("-" * 40)

    # ----------------------------
    # Résumé global
    # ----------------------------
    print("\n===== RÉSUMÉ GLOBAL =====")
    print(f"Images traitées : {len(image_paths)}")
    print(f"Objets détectés au total : {total_objets}")

    if compteur_classes:
        print("\nRépartition par classe :")
        for classe, count in compteur_classes.most_common():
            print(f"  {classe} : {count}")
    else:
        print("Aucun objet détecté sur l'ensemble des images.")

    # ----------------------------
    # Export CSV des résultats détaillés
    # ----------------------------
    csv_path = "resultats_detection.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "classe", "confiance"])
        writer.writeheader()
        writer.writerows(resume_par_image)

    print(f"\nRésultats détaillés exportés dans : {csv_path}")
    print("Les images annotées sont enregistrées dans : runs/detect/predict/")


if __name__ == "__main__":
    main()
