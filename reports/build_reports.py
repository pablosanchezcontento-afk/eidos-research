from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports"
ASSETS = OUT / "assets"
OUT.mkdir(exist_ok=True)
ASSETS.mkdir(exist_ok=True)

GPU_EAGER = json.loads((ROOT / "results/raw/sappho_vs_transformer_10m_5070.json").read_text())
GPU_COMPILE = json.loads((ROOT / "results/raw/sappho_vs_transformer_10m_5070_compile.json").read_text())
CPU = json.loads((ROOT / "results/v0.4/cpu_diagnostic.json").read_text())


def make_charts() -> None:
    names = ["SAPPHO", "Transformer"]
    eager = [r["training_tokens_per_second"] for r in GPU_EAGER["results"]]
    compiled = [r["training_tokens_per_second"] for r in GPU_COMPILE["results"]]
    x = range(len(names))
    fig, ax = plt.subplots(figsize=(7.2, 4.1))
    width = 0.34
    ax.bar([i - width / 2 for i in x], eager, width, label="Sin compile")
    ax.bar([i + width / 2 for i in x], compiled, width, label="Con compile")
    ax.set_xticks(list(x), names)
    ax.set_ylabel("Tokens de entrenamiento por segundo")
    ax.set_title("Resultado real en RTX 5070 - SAPPHO v0.3")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ASSETS / "sappho_gpu_throughput.png", dpi=180)
    plt.close(fig)

    rows = CPU["results"]
    labels = [r["name"].replace("_", " ") for r in rows]
    ratios = [r["ratio_vs_transformer"] for r in rows]
    fig, ax = plt.subplots(figsize=(8.2, 4.3))
    ax.bar(labels, ratios)
    ax.axhline(1.0, linewidth=1)
    ax.set_ylabel("Ratio de throughput frente a Transformer")
    ax.set_title("Diagnostico CPU EIDOS v0.4 - no sustituye la RTX 5070")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(ASSETS / "eidos_cpu_diagnostic.png", dpi=180)
    plt.close(fig)

    model_names = ["Glyph", "Verse", "Epic", "Mythos"]
    params = [11.05, 26.22, 98.48, 173.81]
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.bar(model_names, params)
    ax.set_ylabel("Millones de parametros")
    ax.set_title("Familia EIDOS v0.4")
    fig.tight_layout()
    fig.savefig(ASSETS / "eidos_family.png", dpi=180)
    plt.close(fig)


make_charts()

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="TitleX", parent=styles["Title"], alignment=TA_CENTER, fontSize=25, leading=30, spaceAfter=14))
styles.add(ParagraphStyle(name="SubX", parent=styles["Normal"], alignment=TA_CENTER, fontSize=11, leading=15, textColor=colors.HexColor("#444444"), spaceAfter=16))
styles.add(ParagraphStyle(name="H1X", parent=styles["Heading1"], fontSize=17, leading=21, spaceBefore=10, spaceAfter=7))
styles.add(ParagraphStyle(name="H2X", parent=styles["Heading2"], fontSize=13, leading=17, spaceBefore=8, spaceAfter=5))
styles.add(ParagraphStyle(name="BodyX", parent=styles["BodyText"], fontSize=9.4, leading=13.2, spaceAfter=6))
styles.add(ParagraphStyle(name="SmallX", parent=styles["BodyText"], fontSize=8, leading=10.5, spaceAfter=4))
styles.add(ParagraphStyle(name="Callout", parent=styles["BodyText"], fontSize=10, leading=14, leftIndent=10, rightIndent=10, borderWidth=0.7, borderColor=colors.HexColor("#777777"), borderPadding=8, spaceBefore=7, spaceAfter=9))
styles.add(ParagraphStyle(name="CodeX", parent=styles["Code"], fontName="Courier", fontSize=7.6, leading=10, leftIndent=8, rightIndent=8, borderWidth=0.4, borderColor=colors.HexColor("#bbbbbb"), borderPadding=6, spaceAfter=7))


def esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.grey)
    canvas.drawString(1.55 * cm, 1.0 * cm, "EIDOS v0.4 - registro tecnico - 29 julio 2026")
    canvas.drawRightString(19.4 * cm, 1.0 * cm, f"Pagina {doc.page}")
    canvas.restoreState()


def table(data, widths=None, font_size=7.8):
    cooked = []
    for row in data:
        cooked.append([cell if hasattr(cell, "wrap") else Paragraph(str(cell), styles["SmallX"]) for cell in row])
    t = Table(cooked, colWidths=widths, repeatRows=1, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#222222")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("LEADING", (0, 0), (-1, -1), font_size + 2),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#aaaaaa")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f4f4")]),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return t


def img(path: Path, width_cm: float):
    from reportlab.platypus import Image
    im = Image(str(path))
    ratio = im.imageHeight / im.imageWidth
    im.drawWidth = width_cm * cm
    im.drawHeight = width_cm * cm * ratio
    return im


def build_technical(path: Path) -> None:
    eager_rows = {r["name"]: r for r in GPU_EAGER["results"]}
    comp_rows = {r["name"]: r for r in GPU_COMPILE["results"]}
    story = [
        Spacer(1, 1.0 * cm),
        Paragraph("EIDOS v0.4", styles["TitleX"]),
        Paragraph("Efficient Innovation-Driven Ordered State", styles["SubX"]),
        Paragraph("Rediseño arquitectonico, metodologia de falsacion y registro completo del proyecto NOVA -> ISADORA -> SAPPHO -> EIDOS", styles["SubX"]),
        Spacer(1, 0.3 * cm),
        Paragraph("Veredicto ejecutivo", styles["H1X"]),
        Paragraph(
            "<b>EIDOS no se declara mejor que Transformer todavia.</b> La version v0.3 fue medida en la RTX 5070 y fracaso claramente en eficiencia: incluso compilada alcanzo solo el 48,2% del throughput del Transformer. v0.4 no intenta maquillar ese resultado; cambia la topologia para que la memoria deje de duplicar la atencion, reduce el estado a bajo rango y usa atencion exacta solo como ancla periodica. El paquete bloquea el entrenamiento Mythos hasta que una misma variante gane a todos los baselines en test sellado y sea al menos un 5% mas rapida que Transformer.",
            styles["Callout"],
        ),
        Paragraph("Estado actual", styles["H2X"]),
        table([
            ["Elemento", "Estado comprobado"],
            ["Scan NOVA v0.1", "Bug reproducido y corregido; el usuario verifico la composicion afin frente a un oraculo float64."],
            ["Metodologia v0.2", "Auditoria honesta, pero los BPE estaban practicamente sin aprender y la rejilla de LR tocaba el techo."],
            ["SAPPHO v0.3 en RTX 5070", "12.141 tok/s compilada frente a 25.171 tok/s del Transformer: ratio 0,482."],
            ["EIDOS v0.4", "Implementado y probado localmente; falta ejecutar el benchmark real en la RTX 5070."],
            ["Victoria arquitectonica", "No demostrada. Mythos permanece bloqueado."],
        ], [4.1 * cm, 12.4 * cm]),
        PageBreak(),
    ]

    story += [
        Paragraph("1. Nombre y familia", styles["H1X"]),
        Paragraph(
            "<b>EIDOS</b> es una palabra griega asociada a forma, estructura o esencia. En el proyecto se expande como <b>Efficient Innovation-Driven Ordered State</b>: estado ordenado por escalas temporales, escritura guiada por innovacion y un diseño que prioriza eficiencia medible. No es el nombre de una persona y no fuerza una historia falsa sobre el resultado.",
            styles["BodyX"],
        ),
        table([
            ["Nombre", "Parametros", "Significado metodologico", "Puerta"],
            ["EIDOS Glyph", "11,05M", "La marca minima que comprueba que el instrumento mide y que el candidato aprende.", "GPU + LR + 20 tokens/parametro"],
            ["EIDOS Verse", "26,22M", "Estructura repetible: confirma varias semillas y que la ventaja no era accidental.", "Ganar en calidad por tiempo"],
            ["EIDOS Epic", "98,48M", "Prueba de escala, composicion, codigo y contexto largo.", "Mantener la ventaja"],
            ["EIDOS Mythos", "173,81M", "Sistema completo especializado, no un salto de fe.", "Bloqueado hasta superar todas las puertas"],
        ], [3.0 * cm, 2.3 * cm, 7.0 * cm, 4.2 * cm]),
        Spacer(1, 0.2 * cm),
        img(ASSETS / "eidos_family.png", 13.5),
        Paragraph("La nomenclatura describe el nivel de evidencia, no solo el tamaño del modelo.", styles["SmallX"]),
    ]

    story += [
        PageBreak(),
        Paragraph("2. Cronologia honesta del proyecto", styles["H1X"]),
        table([
            ["Version", "Que aporto", "Que fallo", "Decision"],
            ["NOVA-Lattice v0.1", "ABIM, tests de causalidad, cache, entrenador y una señal de cruce pequeña.", "Baseline con menos LR, ablations con mas parametros y scan silenciosamente corrupto en chunks grandes.", "Invalidar el cruce y corregir el scan."],
            ["ISADORA v0.2", "Scan afin estable, comparaciones mas iguales y dos informes.", "BPE casi no aprendio; 19/19 selecciones de LR tocaron el techo; presupuesto de tokens ridiculo.", "Aceptar que los rankings BPE no significaban nada."],
            ["SAPPHO v0.3", "Atencion como backbone y memoria sidecar conservadora; asserts metodologicos.", "En la RTX 5070 fue entre 2,07x y 2,84x mas lenta que Transformer.", "Matar la topologia sidecar como candidata eficiente."],
            ["EIDOS v0.4", "Memoria de bajo rango que sustituye capas, conv causal barata y anclas periodicas de atencion.", "Calidad y velocidad GPU aun no medidas.", "Ejecutar Glyph; no escalar por entusiasmo."],
        ], [2.6 * cm, 4.6 * cm, 5.0 * cm, 4.3 * cm]),
        Paragraph(
            "El cambio importante no es el nombre. Es que cada version conserva los controles correctos, elimina la deuda demostrada y cambia la arquitectura cuando los datos la contradicen.",
            styles["Callout"],
        ),
    ]

    story += [
        PageBreak(),
        Paragraph("3. Resultado real de la RTX 5070 que obliga al rediseño", styles["H1X"]),
        img(ASSETS / "sappho_gpu_throughput.png", 15.5),
        table([
            ["Modo", "SAPPHO tok/s", "Transformer tok/s", "Ratio", "VRAM SAPPHO", "VRAM Transformer"],
            ["Eager", f"{eager_rows['sappho_lattice']['training_tokens_per_second']:.0f}", f"{eager_rows['transformer_full']['training_tokens_per_second']:.0f}", f"{GPU_EAGER['sappho_throughput_ratio_vs_transformer']:.3f}", f"{eager_rows['sappho_lattice']['peak_allocated_gib']:.3f} GiB", f"{eager_rows['transformer_full']['peak_allocated_gib']:.3f} GiB"],
            ["torch.compile", f"{comp_rows['sappho_lattice']['training_tokens_per_second']:.0f}", f"{comp_rows['transformer_full']['training_tokens_per_second']:.0f}", f"{GPU_COMPILE['sappho_throughput_ratio_vs_transformer']:.3f}", f"{comp_rows['sappho_lattice']['peak_allocated_gib']:.3f} GiB", f"{comp_rows['transformer_full']['peak_allocated_gib']:.3f} GiB"],
        ], [2.5 * cm, 2.6 * cm, 2.8 * cm, 1.6 * cm, 3.0 * cm, 3.0 * cm]),
        Paragraph(
            "La compilacion multiplico SAPPHO por 2,94, pero no resolvio la topologia: cada bloque de fusion pagaba atencion local y memoria completa. La conclusion correcta no es optimizar mas el mismo sidecar, sino eliminar el trabajo duplicado.",
            styles["Callout"],
        ),
    ]

    story += [
        PageBreak(),
        Paragraph("4. Arquitectura EIDOS", styles["H1X"]),
        Paragraph("Macrociclo principal: <b>C -> M -> C -> A</b>.", styles["Callout"]),
        table([
            ["Bloque", "Funcion", "Complejidad temporal", "Motivo"],
            ["C - Causal Conv Mixer", "Convolucion depthwise causal con proyecciones densas y compuerta.", "Lineal en secuencia", "Modela patrones locales sin una matriz de atencion."],
            ["M - Pulse Memory", "Memoria predictiva de bajo rango con escritura por sorpresa y tres escalas.", "Lineal en secuencia", "Conserva estado sin escanear tres copias del ancho completo."],
            ["C - Causal Conv Mixer", "Segundo refinamiento local antes de recuperar globalmente.", "Lineal", "Reduce la frecuencia de atencion sin dejar huecos locales."],
            ["A - Attention Anchor", "GQA causal global con RoPE y QK norm.", "Cuadratica", "Recuperacion exacta periodica y ruta conocida de calidad."],
        ], [2.5 * cm, 6.1 * cm, 3.0 * cm, 4.5 * cm]),
        Paragraph("Cambios frente a SAPPHO", styles["H2X"]),
        table([
            ["SAPPHO v0.3", "EIDOS v0.4"],
            ["Atencion y memoria se calculan juntas en el mismo mixer.", "La memoria sustituye una capa; no duplica atencion."],
            ["Estado en tres bandas para todo el ancho d.", "Estado de rango r=d/8 en los perfiles principales."],
            ["Decays por token, banda y canal.", "Tres decays aprendibles y ordenados, compartidos por canales de rango."],
            ["Scan afin logaritmico con concatenaciones repetidas.", "Scan por chunks con cumsum renormalizado y guardas estaticas compatibles con compile."],
            ["Dos bloques de fusion por cuatro capas.", "Una memoria, dos conv y una ancla global por cuatro capas."],
        ], [8.2 * cm, 8.2 * cm]),
    ]

    story += [
        PageBreak(),
        Paragraph("5. Pulse Memory: mecanismo", styles["H1X"]),
        Paragraph(
            "La memoria recibe una representacion causal filtrada. Proyecta tres vectores de rango reducido: valor, prediccion y control. La innovacion es la diferencia entre el valor actual y la prediccion del paso anterior. Una sorpresa alta puede aumentar el presupuesto total de escritura, pero las tres bandas compiten mediante softmax. La lectura usa una ruta independiente.",
            styles["BodyX"],
        ),
        Paragraph(
            "innovation_t = value_t - prediction_(t-1)<br/>budget_t = sigmoid(B(control_t) + gain * log(1 + RMS(innovation_t)))<br/>write_t = softmax(W(control_t)) * budget_t<br/>state_(t,k) = decay_k * state_(t-1,k) + (1-decay_k) * write_(t,k) * innovation_t<br/>read_t = sum_k softmax(R(control_t + innovation_t))_k * state_(t,k)",
            styles["CodeX"],
        ),
        Paragraph("Por que puede ser mas eficiente", styles["H2X"]),
        table([
            ["Decision", "Efecto"],
            ["Rango d/8", "Reduce proyecciones, activaciones y estado recurrente aproximadamente ocho veces frente al ancho completo."],
            ["Decays escalares por banda", "Evita construir un tensor dinamico [B,T,K,D] de decays y facilita la paralelizacion."],
            ["Atencion cada cuatro capas", "Reduce el coste cuadratico sin eliminar recuperacion exacta."],
            ["Estado constante en generacion", "Cada capa M guarda solo K x r valores y dos tokens de historial convolucional."],
        ], [5.0 * cm, 11.4 * cm]),
        Paragraph(
            "Riesgo: reducir el rango o la frecuencia de atencion puede perder capacidad. Por eso se prueban variantes Core, Wide Memory, Dual Anchor y Pulse; no se fija el ganador por diseño.",
            styles["Callout"],
        ),
    ]

    story += [
        PageBreak(),
        Paragraph("6. Scan rapido y estabilidad numerica", styles["H1X"]),
        Paragraph(
            "NOVA v0.1 usaba cumprod, division y clamp. Cuando el producto caia bajo el clamp, la division utilizaba un denominador artificial y corrompia silenciosamente el estado. ISADORA corrigio el problema con composicion afin asociativa. EIDOS mantiene ese oraculo y añade una ruta mas barata para decays fijos.",
            styles["BodyX"],
        ),
        Paragraph(
            "Dentro de cada chunk y para decay fijo a:<br/>h_i = a^(i+1) * (h_inicio + sum_(j<=i) b_j / a^(j+1))",
            styles["CodeX"],
        ),
        table([
            ["Control", "Implementacion"],
            ["Sin clamp", "No se sustituye ningun producto por un valor artificial."],
            ["Chunk maximo 128", "La restriccion es estatica y no introduce sincronizacion GPU-CPU dentro de torch.compile."],
            ["Decay minimo 0,85", "A chunk 128, la potencia minima permanece por encima de 1e-9 en float32."],
            ["Oraculo secuencial", "Tests comparan chunks 1, 4, 16, 32, 64 y 128 contra recurrencia exacta."],
            ["Backward", "Gradientes de decays e inyecciones verificados como finitos."],
            ["Cache", "Forward token a token coincide con forward completo."],
        ], [4.4 * cm, 12.0 * cm]),
        Paragraph("La suite nueva contiene 16 tests y todos pasan en este entorno.", styles["Callout"]),
    ]

    story += [
        PageBreak(),
        Paragraph("7. Comparacion de parametros y diagnostico local", styles["H1X"]),
        table([
            ["Modelo Glyph", "Parametros", "Gap vs EIDOS Core", "FFN", "Rango memoria"],
            ["EIDOS Core", "11.049.963", "0,000%", "640", "32"],
            ["EIDOS Wide Memory", "11.058.379", "0,076%", "632", "64"],
            ["EIDOS Dual Anchor", "11.048.171", "0,016%", "640", "32"],
            ["EIDOS Pulse", "11.049.430", "0,005%", "672", "32"],
            ["Transformer", "11.045.120", "0,044%", "608", "-"],
            ["Conv Striped", "11.050.496", "0,005%", "608", "-"],
            ["Recurrent Striped", "11.057.350", "0,067%", "680", "32"],
        ], [4.1 * cm, 3.0 * cm, 3.0 * cm, 2.2 * cm, 3.1 * cm]),
        img(ASSETS / "eidos_cpu_diagnostic.png", 16.0),
        Paragraph(
            "En el diagnostico CPU de secuencia 512, EIDOS Core marco 1,29x el throughput del Transformer y Conv Striped 1,28x. Esto solo indica que el rediseño ya no tiene la penalizacion estructural de SAPPHO. No demuestra rendimiento CUDA ni calidad; el orden puede cambiar completamente en la RTX 5070.",
            styles["Callout"],
        ),
    ]

    story += [
        PageBreak(),
        Paragraph("8. Metodologia que impide otra falsa victoria", styles["H1X"]),
        table([
            ["Fallo historico", "Barrera v0.4"],
            ["Resultado peor que uniforme incluido en tablas", "Cada run calcula log2(vocab); si no lo supera por margen, status de aprendizaje falla."],
            ["Mejor LR en el techo", "select_lrs.py bloquea la arquitectura y propone ampliar la rejilla."],
            ["1-3% del corpus y menos de 0,01 tokens/parametro", "train.py aborta por defecto bajo 20 tokens/parametro; solo smoke exige override explicito."],
            ["Parametros aproximados mal declarados", "Matching calculado y aborta si el gap supera 0,5%."],
            ["Test usado para elegir LR", "Calibracion usa validacion; el test solo se abre al final de confirmacion."],
            ["Arquitectura elegida por loss sin coste", "La puerta final exige calidad sellada y throughput real de GPU para la misma variante."],
            ["Solo Transformer como rival", "Se incluyen Transformer, Conv Striped y Recurrent Striped; gana el mejor baseline medido."],
        ], [5.2 * cm, 11.2 * cm]),
        Paragraph("Puerta Mythos", styles["H2X"]),
        Paragraph(
            "Una variante EIDOS solo desbloquea Mythos si: (1) tiene al menos tres semillas; (2) todas aprenden por debajo de uniforme; (3) usa al menos 20 tokens por parametro; (4) gap de parametros <=0,5%; (5) mejora al mejor baseline en test sellado por al menos 0,5%; y (6) supera a Transformer en throughput por al menos 5% en la RTX 5070.",
            styles["Callout"],
        ),
    ]

    story += [
        PageBreak(),
        Paragraph("9. Datos y tokenizer", styles["H1X"]),
        Paragraph(
            "El corpus proxy de 207k tokens queda relegado a tests de software. La comparacion Glyph exige texto real, BPE de 32.768 piezas y splits por documento. pack_corpus.py ahora elimina duplicados exactos antes del split y genera train, validation y test mediante hash determinista.",
            styles["BodyX"],
        ),
        table([
            ["Etapa", "Control"],
            ["Descarga", "Fuentes separadas para web educativo, español y codigo; registrar origen y licencia."],
            ["Deduplicacion", "Blake2b sobre texto normalizado antes de tokenizar."],
            ["Split", "Hash de documento; test y validation no son ventanas aleatorias del mismo archivo."],
            ["Tokenizer", "BPE byte-level 32k entrenado desde cero sobre la mezcla final."],
            ["Empaquetado", "uint16 mientras vocab <=65.535; manifiesto con documentos y tokens por split."],
            ["Contaminacion", "Benchmarks privados deben construirse antes de descargar el corpus final."],
        ], [4.0 * cm, 12.4 * cm]),
        Paragraph("Comandos base", styles["H2X"]),
        Paragraph(
            "python fetch_corpus.py --preset web --out corpus/web.jsonl --max-gb 4<br/>"
            "python fetch_corpus.py --preset spanish_web --out corpus/es.jsonl --max-gb 2<br/>"
            "python fetch_corpus.py --preset code --out corpus/code.jsonl --max-gb 2<br/>"
            "python train_tokenizer.py corpus --vocab-size 32768 --out data/tokenizer.json<br/>"
            "python pack_corpus.py corpus --tokenizer data/tokenizer.json --train-out data/train.bin --val-out data/val.bin --test-out data/test.bin",
            styles["CodeX"],
        ),
    ]

    story += [
        PageBreak(),
        Paragraph("10. Orden exacto de ejecucion", styles["H1X"]),
        table([
            ["Paso", "Accion", "Tiempo esperado", "Decision"],
            ["1", "scripts/benchmark_eidos_5070.ps1", "Minutos", "Eliminar variantes que no lleguen a 1,05x Transformer."],
            ["2", "Transformer Glyph de cordura", "Una corrida", "Si no aprende limpiamente, el instrumento sigue roto."],
            ["3", "Calibracion LR a 2 tokens/parametro", "Varias corridas cortas", "Solo elegir LR; no declarar ganador."],
            ["4", "Extender rejilla si el optimo toca borde", "Segun resultado", "Nadie avanza con LR truncado."],
            ["5", "Confirmacion a 20 tokens/parametro y 3 semillas", "Horas", "Comparar test sellado y tiempo real."],
            ["6", "evaluate_gate.py", "Segundos", "Desbloquear o bloquear Verse/Mythos sin negociacion."],
        ], [1.2 * cm, 7.1 * cm, 3.2 * cm, 5.0 * cm]),
        Paragraph("Primer comando", styles["H2X"]),
        Paragraph("powershell -ExecutionPolicy Bypass -File .\\scripts\\benchmark_eidos_5070.ps1", styles["CodeX"]),
        Paragraph(
            "La corrida nueva compara siete modelos emparejados: cuatro variantes EIDOS y tres baselines fuertes. El random loss se etiqueta expresamente como no interpretable; el script solo decide velocidad y VRAM.",
            styles["Callout"],
        ),
    ]

    story += [
        PageBreak(),
        Paragraph("11. Registro de acciones realizadas en esta conversacion", styles["H1X"]),
    ]
    action_lines = (ROOT / "ACTION_LOG.md").read_text(encoding="utf-8").splitlines()
    for line in action_lines:
        if line.startswith("#"):
            story.append(Paragraph(esc(line.lstrip("# ")), styles["H2X"]))
        elif line.startswith("- "):
            story.append(Paragraph("- " + esc(line[2:]), styles["BodyX"]))
        elif line.strip():
            story.append(Paragraph(esc(line), styles["BodyX"]))

    story += [
        Paragraph("12. Conclusion", styles["H1X"]),
        Paragraph(
            "EIDOS v0.4 es una arquitectura nueva y mas plausible para el objetivo que SAPPHO: elimina computo duplicado, reduce el estado, mantiene atencion exacta periodica y endurece el protocolo. Pero la obligacion de ser mejor no se satisface escribiendola en un PDF. Se satisface cuando la misma variante supera al mejor baseline en calidad sellada, velocidad y reproducibilidad. Hasta entonces, EIDOS Glyph es un candidato; EIDOS Mythos no existe como modelo aprobado.",
            styles["Callout"],
        ),
    ]

    SimpleDocTemplate(
        str(path), pagesize=A4, rightMargin=1.55 * cm, leftMargin=1.55 * cm,
        topMargin=1.55 * cm, bottomMargin=1.45 * cm,
        title="EIDOS v0.4 Informe Tecnico",
        author="Proyecto personal de Pablo",
    ).build(story, onFirstPage=footer, onLaterPages=footer)


def build_plain(path: Path) -> None:
    story = [
        Spacer(1, 1.0 * cm),
        Paragraph("EIDOS v0.4", styles["TitleX"]),
        Paragraph("La version para entenderlo sin tragarte veinte paginas de matematicas", styles["SubX"]),
        Paragraph("Lo mas importante", styles["H1X"]),
        Paragraph(
            "<b>No, todavia no tenemos un modelo mejor que Transformer.</b> Lo que si tenemos ahora es un diseño distinto que ataca el fallo real descubierto en tu RTX 5070. SAPPHO hacia atencion y memoria a la vez, por eso incluso compilada iba a menos de la mitad de velocidad. EIDOS deja de pagar dos veces por la misma capa.",
            styles["Callout"],
        ),
        img(ASSETS / "sappho_gpu_throughput.png", 15.0),
        Paragraph("El benchmark anterior fue util precisamente porque nos obligo a tirar una idea que no daba la talla.", styles["SmallX"]),
        PageBreak(),
        Paragraph("El nombre", styles["H1X"]),
        Paragraph(
            "<b>EIDOS</b> significa forma o esencia. La sigla del proyecto es <b>Efficient Innovation-Driven Ordered State</b>: una memoria eficiente que guarda cambios utiles en escalas ordenadas. Ya no hay nombres de personas ni acronimos pegados con cinta aislante.",
            styles["BodyX"],
        ),
        table([
            ["Modelo", "Que significa"],
            ["Glyph", "La prueba pequeña que demuestra que el instrumento funciona."],
            ["Verse", "La confirmacion repetida: no basta una corrida afortunada."],
            ["Epic", "La prueba grande de escala, codigo y razonamiento verificable."],
            ["Mythos", "El modelo serio de 174M; esta bloqueado hasta ganar de verdad."],
        ], [4.0 * cm, 12.4 * cm]),
        Paragraph("Como funciona", styles["H1X"]),
        Paragraph(
            "Cada grupo de cuatro capas hace esto: <b>convolucion barata -> memoria -> convolucion barata -> atencion global</b>. El Transformer normal hace atencion en todas las capas. EIDOS intenta ahorrar ahi sin quedarse ciego: cada cuatro capas usa una atencion completa para recuperar cualquier parte del contexto.",
            styles["Callout"],
        ),
        Paragraph(
            "La memoria tampoco guarda todo. Primero intenta predecir lo que viene. Solo escribe con fuerza cuando el token aporta algo que no esperaba. Ademas, guarda esa novedad en una memoria rapida, una media y una lenta.",
            styles["BodyX"],
        ),
        PageBreak(),
        Paragraph("Que se ha cambiado de verdad", styles["H1X"]),
        table([
            ["Antes", "Ahora"],
            ["Memoria de ancho completo", "Memoria de rango reducido: 32 canales internos en Glyph frente a 256 del modelo."],
            ["Atencion y memoria juntas", "La memoria sustituye una capa; no se suma encima."],
            ["Solo Transformer como rival", "Transformer, convolucional y recurrente compiten en el mismo test."],
            ["Podia colarse un modelo peor que azar", "La corrida falla automaticamente si no supera la distribucion uniforme."],
            ["El mejor LR podia estar en el borde", "El selector obliga a ampliar la rejilla."],
            ["Pocos tokens se vendian como resultado", "El entrenamiento serio exige 20 tokens por parametro."],
        ], [7.9 * cm, 8.5 * cm]),
        Paragraph("¿Pinta mejor?", styles["H1X"]),
        img(ASSETS / "eidos_cpu_diagnostic.png", 15.5),
        Paragraph(
            "En una prueba de CPU, EIDOS Core fue aproximadamente 1,29 veces mas rapido que el Transformer. Eso es una buena señal porque SAPPHO era claramente mas lento. Pero no cuenta como victoria: la RTX 5070 puede ordenar los modelos de otra forma y aun falta comprobar quien aprende mejor.",
            styles["Callout"],
        ),
        PageBreak(),
        Paragraph("Que significa ganar", styles["H1X"]),
        Paragraph(
            "No me vale que empate ni que gane por una milesima en una corrida. La misma version tiene que:",
            styles["BodyX"],
        ),
        table([
            ["Condicion", "Minimo"],
            ["Velocidad", "Al menos 5% mas rapida que Transformer en tu RTX 5070."],
            ["Calidad", "Al menos 0,5% menos error que el mejor entre Transformer, conv y recurrente."],
            ["Repeticion", "Tres semillas."],
            ["Datos", "20 tokens por parametro sobre BPE real."],
            ["Justicia", "Diferencia de parametros inferior al 0,5%."],
            ["Test", "Sellado hasta el final."],
        ], [4.0 * cm, 12.4 * cm]),
        Paragraph(
            "Si no pasa todo a la vez, Mythos sigue cerrado. Nada de cambiar la regla cuando el resultado nos cae mal; ese deporte ya lo practican demasiados benchmarks de Twitter.",
            styles["Callout"],
        ),
        Paragraph("Que haces ahora", styles["H1X"]),
        Paragraph("Abre PowerShell dentro de la carpeta y ejecuta:", styles["BodyX"]),
        Paragraph("powershell -ExecutionPolicy Bypass -File .\\scripts\\benchmark_eidos_5070.ps1", styles["CodeX"]),
        Paragraph(
            "Ese benchmark prueba EIDOS Core, Wide Memory, Dual Anchor y Pulse contra Transformer, Conv Striped y Recurrent Striped. Cuando tengamos el JSON sabremos si el rediseño realmente arreglo la eficiencia. Solo entonces hacemos la matriz de aprendizaje.",
            styles["BodyX"],
        ),
        Paragraph("Veredicto final", styles["H1X"]),
        Paragraph(
            "EIDOS es bastante mas serio que SAPPHO porque nace de un resultado real de tu GPU y no de una grafica bonita. Aun no es el campeon. Es el primer candidato de este proyecto diseñado para tener una posibilidad razonable de ser mas rapido sin renunciar a memoria y recuperacion global.",
            styles["Callout"],
        ),
    ]
    SimpleDocTemplate(
        str(path), pagesize=A4, rightMargin=1.65 * cm, leftMargin=1.65 * cm,
        topMargin=1.65 * cm, bottomMargin=1.45 * cm,
        title="EIDOS v0.4 Explicado Facil",
        author="Proyecto personal de Pablo",
    ).build(story, onFirstPage=footer, onLaterPages=footer)


build_technical(OUT / "EIDOS_v0.4_Informe_Tecnico.pdf")
build_plain(OUT / "EIDOS_v0.4_Explicado_Facil.pdf")
print("reports built")
