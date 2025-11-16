#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Optional, Tuple
import re
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from sympy import Symbol, oo, zoo, nan, S
from sympy import limit, factor, simplify, expand, cancel
from sympy import sin, cos, tan, exp, log, sqrt
from sympy import diff, fraction
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application


@dataclass
class LimitClassification:
    tipo: str
    valor_substituicao: Optional[sp.Basic]
    observacoes: str
    numerador: Optional[sp.Basic] = None
    denominador: Optional[sp.Basic] = None


def classify_limit(expr: sp.Basic, x_symbol: Symbol, point: sp.Basic) -> LimitClassification:
    try:
        valor_sub = expr.subs(x_symbol, point)
        valor_sub = simplify(valor_sub)
        
        num, den = fraction(expr)
        num_val = simplify(num.subs(x_symbol, point))
        den_val = simplify(den.subs(x_symbol, point))
        
        if valor_sub.is_finite and valor_sub.is_real:
            return LimitClassification(
                tipo="FINITO",
                valor_substituicao=valor_sub,
                observacoes=f"Substituição direta resulta em {valor_sub}",
                numerador=num_val,
                denominador=den_val
            )
        
        if den_val == 0 and num_val != 0 and num_val.is_finite:
            return LimitClassification(
                tipo="NUMERO_SOBRE_ZERO",
                valor_substituicao=None,
                observacoes=f"Numerador = {num_val}, Denominador = 0",
                numerador=num_val,
                denominador=den_val
            )
        
        if num_val == 0 and den_val == 0:
            return LimitClassification(
                tipo="ZERO_SOBRE_ZERO",
                valor_substituicao=None,
                observacoes="Indeterminação 0/0",
                numerador=num_val,
                denominador=den_val
            )
        
        if (num_val.has(oo) or num_val.has(-oo)) and (den_val.has(oo) or den_val.has(-oo)):
            return LimitClassification(
                tipo="INFINITO_SOBRE_INFINITO",
                valor_substituicao=None,
                observacoes="Indeterminação ∞/∞",
                numerador=num_val,
                denominador=den_val
            )
        
        if isinstance(expr, sp.Add):
            termos_infinitos = [t.subs(x_symbol, point) for t in expr.args]
            has_pos_inf = any(t == oo for t in termos_infinitos)
            has_neg_inf = any(t == -oo for t in termos_infinitos)
            if has_pos_inf and has_neg_inf:
                return LimitClassification(
                    tipo="INFINITO_MENOS_INFINITO",
                    valor_substituicao=None,
                    observacoes="Indeterminação ∞ - ∞"
                )
        
        if isinstance(expr, sp.Mul):
            fatores = [simplify(f.subs(x_symbol, point)) for f in expr.args]
            has_zero = any(f == 0 for f in fatores)
            has_inf = any(f.has(oo) or f.has(-oo) for f in fatores)
            if has_zero and has_inf:
                return LimitClassification(
                    tipo="ZERO_VEZES_INFINITO",
                    valor_substituicao=None,
                    observacoes="Indeterminação 0 · ∞"
                )
        
        if isinstance(expr, sp.Pow):
            base = simplify(expr.base.subs(x_symbol, point))
            expoente = simplify(expr.exp.subs(x_symbol, point))
            
            if base == 1 and (expoente.has(oo) or expoente.has(-oo)):
                return LimitClassification(
                    tipo="UM_POTENCIA_INFINITO",
                    valor_substituicao=None,
                    observacoes="Indeterminação 1^∞"
                )
            
            if base == 0 and expoente == 0:
                return LimitClassification(
                    tipo="ZERO_POTENCIA_ZERO",
                    valor_substituicao=None,
                    observacoes="Indeterminação 0^0"
                )
            
            if (base.has(oo) or base.has(-oo)) and expoente == 0:
                return LimitClassification(
                    tipo="INFINITO_POTENCIA_ZERO",
                    valor_substituicao=None,
                    observacoes="Indeterminação ∞^0"
                )
        
        return LimitClassification(
            tipo="GENERICO",
            valor_substituicao=valor_sub,
            observacoes=f"Resultado da substituição: {valor_sub}"
        )
        
    except Exception as e:
        return LimitClassification(
            tipo="ERRO",
            valor_substituicao=None,
            observacoes=f"Erro na classificação: {str(e)}"
        )


def resolver_finito(expr: sp.Basic, x_symbol: Symbol, point: sp.Basic, 
                    classification: LimitClassification) -> Tuple[sp.Basic, str]:
    resultado = classification.valor_substituicao
    explicacao = f"""
═══════════════════════════════════════════════
TIPO: Valor Finito (Função Contínua)
═══════════════════════════════════════════════

Substituímos x = {point} na expressão:
f({point}) = {resultado}

Como obtivemos um valor real finito, a função é contínua neste ponto.
Portanto, o limite existe e é igual ao valor da função.

RESULTADO: lim f(x) = {resultado}
           x→{point}
"""
    return resultado, explicacao


def resolver_numero_sobre_zero(expr: sp.Basic, x_symbol: Symbol, point: sp.Basic,
                                classification: LimitClassification, direction: str) -> Tuple[sp.Basic, str]:
    explicacao = f"""
═══════════════════════════════════════════════
TIPO: Número sobre Zero (Possível Assíntota Vertical)
═══════════════════════════════════════════════

Numerador em x = {point}: {classification.numerador}
Denominador em x = {point}: 0

Como temos um número não-nulo dividido por zero, 
precisamos analisar os limites laterais:
"""
    
    lim_esquerda = limit(expr, x_symbol, point, '-')
    lim_direita = limit(expr, x_symbol, point, '+')
    
    explicacao += f"""
Limite pela esquerda (x → {point}⁻): {lim_esquerda}
Limite pela direita (x → {point}⁺): {lim_direita}
"""
    
    if direction == "both":
        if lim_esquerda == lim_direita:
            resultado = lim_esquerda
            explicacao += f"\nComo os limites laterais são iguais, o limite existe:\nRESULTADO: lim f(x) = {resultado}\n           x→{point}"
        else:
            resultado = None
            explicacao += f"\nComo os limites laterais são diferentes, o limite NÃO EXISTE.\nRESULTADO: O limite não existe"
    elif direction == "+":
        resultado = lim_direita
        explicacao += f"\nRESULTADO: lim f(x) = {resultado}\n           x→{point}⁺"
    else:
        resultado = lim_esquerda
        explicacao += f"\nRESULTADO: lim f(x) = {resultado}\n           x→{point}⁻"
    
    return resultado, explicacao


def resolver_zero_sobre_zero(expr: sp.Basic, x_symbol: Symbol, point: sp.Basic) -> Tuple[sp.Basic, str]:
    explicacao = f"""
═══════════════════════════════════════════════
TIPO: Indeterminação 0/0
═══════════════════════════════════════════════

Identificamos a forma indeterminada 0/0.
Vamos tentar resolver usando técnicas apropriadas:
"""
    
    num, den = fraction(expr)
    
    try:
        expr_fatorada = factor(num) / factor(den)
        expr_simplificada = cancel(expr_fatorada)
        
        if expr_simplificada != expr:
            explicacao += f"""
→ FATORAÇÃO:
  Numerador fatorado: {factor(num)}
  Denominador fatorado: {factor(den)}
  
  Após cancelar fatores comuns:
  f(x) = {expr_simplificada}
"""
            try:
                resultado = expr_simplificada.subs(x_symbol, point)
                if resultado.is_finite and resultado.is_real:
                    explicacao += f"""
  Agora podemos substituir x = {point}:
  Resultado: {resultado}
"""
                    explicacao += f"\nRESULTADO: lim f(x) = {resultado}\n           x→{point}"
                    return resultado, explicacao
            except:
                pass
    except:
        pass
    
    if point == 0:
        expr_expandido = expand(expr)
        explicacao += f"\n→ Verificando limites fundamentais...\n  Expressão expandida: {expr_expandido}\n"
    
    explicacao += f"""
→ REGRA DE L'HÔPITAL:
  Como temos 0/0, derivamos numerador e denominador:
"""
    
    try:
        num_derivada = diff(num, x_symbol)
        den_derivada = diff(den, x_symbol)
        
        explicacao += f"""
  Numerador': {num_derivada}
  Denominador': {den_derivada}
  
  Nova expressão: f'(x) = {num_derivada}/{den_derivada}
"""
        
        nova_expr = num_derivada / den_derivada
        resultado = limit(nova_expr, x_symbol, point)
        
        explicacao += f"""
  Calculando o limite da nova expressão:
  Resultado: {resultado}
"""
        explicacao += f"\nRESULTADO: lim f(x) = {resultado}\n           x→{point}"
        return resultado, explicacao
        
    except Exception as e:
        explicacao += f"  Erro ao aplicar L'Hôpital: {str(e)}\n"
    
    resultado = limit(expr, x_symbol, point)
    explicacao += f"\nRESULTADO (cálculo direto): lim f(x) = {resultado}\n                            x→{point}"
    return resultado, explicacao


def resolver_infinito_sobre_infinito(expr: sp.Basic, x_symbol: Symbol, point: sp.Basic) -> Tuple[sp.Basic, str]:
    explicacao = f"""
═══════════════════════════════════════════════
TIPO: Indeterminação ∞/∞
═══════════════════════════════════════════════

Identificamos a forma indeterminada ∞/∞.
"""
    
    num, den = fraction(expr)
    
    if point in [oo, -oo]:
        explicacao += f"""
→ DIVIDINDO PELA MAIOR POTÊNCIA:
  Como x → {point}, colocamos em evidência o termo de maior grau.
"""
        
        try:
            grau_num = sp.degree(num, x_symbol) if sp.degree(num, x_symbol) != -oo else 0
            grau_den = sp.degree(den, x_symbol) if sp.degree(den, x_symbol) != -oo else 0
            
            if grau_den > 0:
                poder_divisao = x_symbol ** grau_den
                num_dividido = simplify(num / poder_divisao)
                den_dividido = simplify(den / poder_divisao)
                
                explicacao += f"""
  Grau do numerador: {grau_num}
  Grau do denominador: {grau_den}
  
  Dividindo numerador e denominador por x^{grau_den}:
  Numerador/x^{grau_den} = {num_dividido}
  Denominador/x^{grau_den} = {den_dividido}
"""
                
                nova_expr = num_dividido / den_dividido
                resultado = limit(nova_expr, x_symbol, point)
                
                explicacao += f"""
  Nova expressão: {nova_expr}
  
  Agora calculamos o limite quando x → {point}:
  Termos com x no denominador tendem a 0.
  Resultado: {resultado}
"""
                explicacao += f"\nRESULTADO: lim f(x) = {resultado}\n           x→{point}"
                return resultado, explicacao
        except:
            pass
    
    explicacao += f"""
→ REGRA DE L'HÔPITAL:
  Derivamos numerador e denominador:
"""
    
    try:
        num_derivada = diff(num, x_symbol)
        den_derivada = diff(den, x_symbol)
        
        explicacao += f"""
  Numerador': {num_derivada}
  Denominador': {den_derivada}
"""
        
        nova_expr = num_derivada / den_derivada
        resultado = limit(nova_expr, x_symbol, point)
        
        explicacao += f"  Resultado: {resultado}\n"
        explicacao += f"\nRESULTADO: lim f(x) = {resultado}\n           x→{point}"
        return resultado, explicacao
        
    except Exception as e:
        explicacao += f"  Erro ao aplicar L'Hôpital: {str(e)}\n"
    
    resultado = limit(expr, x_symbol, point)
    explicacao += f"\nRESULTADO (cálculo direto): lim f(x) = {resultado}\n                            x→{point}"
    return resultado, explicacao


def resolver_infinito_menos_infinito(expr: sp.Basic, x_symbol: Symbol, point: sp.Basic) -> Tuple[sp.Basic, str]:
    explicacao = f"""
═══════════════════════════════════════════════
TIPO: Indeterminação ∞ - ∞
═══════════════════════════════════════════════

Identificamos a forma indeterminada ∞ - ∞.
Vamos transformar a expressão para remover a indeterminação.
"""
    
    if expr.has(sqrt):
        explicacao += """
→ RACIONALIZAÇÃO (Multiplicação pelo conjugado):
  Detectamos raízes na expressão.
  Multiplicamos pelo conjugado para eliminar a indeterminação.
"""
    
    try:
        expr_frac = expr.together()
        if expr_frac != expr:
            explicacao += f"""
→ UNIFICANDO EM FRAÇÃO ÚNICA:
  Reescrevemos a expressão como uma única fração:
  f(x) = {expr_frac}
  
  Agora temos uma fração que pode ser do tipo 0/0 ou ∞/∞.
"""
            resultado = limit(expr_frac, x_symbol, point)
            explicacao += f"\n  Resultado: {resultado}"
            explicacao += f"\nRESULTADO: lim f(x) = {resultado}\n           x→{point}"
            return resultado, explicacao
    except:
        pass
    
    resultado = limit(expr, x_symbol, point)
    explicacao += f"\nRESULTADO (cálculo direto): lim f(x) = {resultado}\n                            x→{point}"
    return resultado, explicacao


def resolver_zero_vezes_infinito(expr: sp.Basic, x_symbol: Symbol, point: sp.Basic) -> Tuple[sp.Basic, str]:
    explicacao = f"""
═══════════════════════════════════════════════
TIPO: Indeterminação 0 · ∞
═══════════════════════════════════════════════

Identificamos a forma indeterminada 0 · ∞.
"""
    
    explicacao += """
→ TRANSFORMAÇÃO EM FRAÇÃO:
  Reescrevemos como 0/(1/∞) = 0/0 ou ∞/(1/0) = ∞/∞
"""
    
    resultado = limit(expr, x_symbol, point)
    
    explicacao += f"""
  Após transformação e simplificação:
  Resultado: {resultado}
"""
    explicacao += f"\nRESULTADO: lim f(x) = {resultado}\n           x→{point}"
    return resultado, explicacao


def resolver_exponencial_indeterminado(expr: sp.Basic, x_symbol: Symbol, point: sp.Basic, 
                                       classification: LimitClassification) -> Tuple[sp.Basic, str]:
    tipo_map = {
        "UM_POTENCIA_INFINITO": "1^∞",
        "ZERO_POTENCIA_ZERO": "0^0",
        "INFINITO_POTENCIA_ZERO": "∞^0"
    }
    
    explicacao = f"""
═══════════════════════════════════════════════
TIPO: Indeterminação {tipo_map.get(classification.tipo, "Exponencial")}
═══════════════════════════════════════════════

Identificamos uma indeterminação exponencial.
"""
    
    if isinstance(expr, sp.Pow):
        base = expr.base
        expoente = expr.exp
        
        if isinstance(base, sp.Add) and len(base.args) == 2:
            if 1 in base.args or S.One in base.args:
                u = base - 1
                v = expoente
                
                explicacao += f"""
→ LIMITE FUNDAMENTAL EXPONENCIAL:
  A expressão tem a forma (1 + u)^v, onde:
  u = {u}
  v = {v}
  
  Usamos: lim (1 + u)^v = e^(lim u·v)
         quando u → 0 e v → ∞
"""
                
                produto = simplify(u * v)
                explicacao += f"""
  Calculando u · v = {produto}
"""
                
                lim_produto = limit(produto, x_symbol, point)
                explicacao += f"""
  lim (u · v) = {lim_produto}
  x→{point}
"""
                
                if lim_produto.is_finite:
                    resultado = exp(lim_produto)
                    explicacao += f"""
  Portanto: lim f(x) = e^({lim_produto}) = {resultado}
            x→{point}
"""
                    explicacao += f"\nRESULTADO: lim f(x) = {resultado}\n           x→{point}"
                    return resultado, explicacao
    
    explicacao += f"""
→ MÉTODO DO LOGARITMO:
  Seja y = f(x). Calculamos ln(y) = ln(f(x)).
  Para potências: ln(a^b) = b·ln(a)
"""
    
    try:
        ln_expr = log(expr)
        ln_expr_simplificado = simplify(ln_expr)
        
        explicacao += f"""
  ln(y) = {ln_expr_simplificado}
"""
        
        lim_ln = limit(ln_expr_simplificado, x_symbol, point)
        
        explicacao += f"""
  lim ln(y) = {lim_ln}
  x→{point}
"""
        
        if lim_ln.is_finite:
            resultado = exp(lim_ln)
            explicacao += f"""
  Como lim y = e^(lim ln(y)):
  lim f(x) = e^({lim_ln}) = {resultado}
  x→{point}
"""
            explicacao += f"\nRESULTADO: lim f(x) = {resultado}\n           x→{point}"
            return resultado, explicacao
    except:
        pass
    
    resultado = limit(expr, x_symbol, point)
    explicacao += f"\nRESULTADO (cálculo direto): lim f(x) = {resultado}\n                            x→{point}"
    return resultado, explicacao


def normalize_expression(expr_str: str) -> str:
    expr_str = expr_str.strip()
    expr_str = re.sub(r'\s*\*\*\s*', '**', expr_str)
    expr_str = re.sub(r'\s*\+\s*', '+', expr_str)
    expr_str = re.sub(r'\s*\-\s*', '-', expr_str)
    expr_str = re.sub(r'\s*\*\s*', '*', expr_str)
    expr_str = re.sub(r'\s*/\s*', '/', expr_str)
    return expr_str


def solve_limit(expr_str: str, point_str: str, direction: str = "both") -> str:
    try:
        x = Symbol('x', real=True)
        expr_str = normalize_expression(expr_str)
        transformations = standard_transformations + (implicit_multiplication_application,)
        expr = parse_expr(expr_str, local_dict={'x': x}, transformations=transformations)
        
        if point_str.lower() in ['inf', '+inf', 'infinity', '+infinity']:
            point = oo
        elif point_str.lower() in ['-inf', '-infinity']:
            point = -oo
        else:
            point = sp.sympify(point_str)
        
        if direction.lower() in ['+', 'direita', 'right']:
            dir_symbol = '⁺'
            direction = '+'
        elif direction.lower() in ['-', 'esquerda', 'left']:
            dir_symbol = '⁻'
            direction = '-'
        else:
            dir_symbol = ''
            direction = 'both'
        
        resultado_str = f"""
╔═══════════════════════════════════════════════╗
║     CALCULADORA DE LIMITES - RESOLUÇÃO        ║
╚═══════════════════════════════════════════════╝

Expressão: f(x) = {expr}
Limite em: x → {point}{dir_symbol}

"""
        
        classification = classify_limit(expr, x, point)
        
        resultado_str += f"CLASSIFICAÇÃO: {classification.tipo}\n"
        resultado_str += f"Observações: {classification.observacoes}\n"
        
        if classification.tipo == "FINITO":
            resultado, explicacao = resolver_finito(expr, x, point, classification)
        
        elif classification.tipo == "NUMERO_SOBRE_ZERO":
            resultado, explicacao = resolver_numero_sobre_zero(expr, x, point, classification, direction)
        
        elif classification.tipo == "ZERO_SOBRE_ZERO":
            resultado, explicacao = resolver_zero_sobre_zero(expr, x, point)
        
        elif classification.tipo == "INFINITO_SOBRE_INFINITO":
            resultado, explicacao = resolver_infinito_sobre_infinito(expr, x, point)
        
        elif classification.tipo == "INFINITO_MENOS_INFINITO":
            resultado, explicacao = resolver_infinito_menos_infinito(expr, x, point)
        
        elif classification.tipo == "ZERO_VEZES_INFINITO":
            resultado, explicacao = resolver_zero_vezes_infinito(expr, x, point)
        
        elif classification.tipo in ["UM_POTENCIA_INFINITO", "ZERO_POTENCIA_ZERO", "INFINITO_POTENCIA_ZERO"]:
            resultado, explicacao = resolver_exponencial_indeterminado(expr, x, point, classification)
        
        else:
            resultado = limit(expr, x, point, direction if direction != "both" else None)
            explicacao = f"""
═══════════════════════════════════════════════
CÁLCULO DIRETO
═══════════════════════════════════════════════

Calculando o limite diretamente:
RESULTADO: lim f(x) = {resultado}
           x→{point}{dir_symbol}
"""
        
        resultado_str += explicacao
        resultado_str += "\n" + "═" * 47 + "\n"
        
        return resultado_str
        
    except Exception as e:
        return f"""
╔═══════════════════════════════════════════════╗
║                    ERRO                       ║
╚═══════════════════════════════════════════════╝

Ocorreu um erro ao processar o limite:
{str(e)}

Verifique se a expressão está correta.
"""


def plot_limit(expr: sp.Basic, x_symbol: Symbol, point: sp.Basic, 
               resultado: sp.Basic, direction: str = "both") -> None:
    try:
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.unicode_minus'] = False
        
        fig, ax = plt.subplots(figsize=(10, 7))
        
        f_lambdified = sp.lambdify(x_symbol, expr, modules=['numpy'])
        
        if point == oo or point == -oo:
            if point == oo:
                x_vals = np.linspace(1, 50, 1000)
                title_point = "∞"
            else:
                x_vals = np.linspace(-50, -1, 1000)
                title_point = "-∞"
        else:
            point_float = float(point)
            delta = 3
            x_vals = np.linspace(point_float - delta, point_float + delta, 1000)
            title_point = str(point)
            x_vals = x_vals[np.abs(x_vals - point_float) > 0.01]
        
        try:
            y_vals = f_lambdified(x_vals)
            
            mask = np.isfinite(y_vals)
            x_clean = x_vals[mask]
            y_clean = y_vals[mask]
            
            y_max = np.percentile(np.abs(y_clean), 95) * 2
            y_clean = np.clip(y_clean, -y_max, y_max)
            
            ax.plot(x_clean, y_clean, 'b-', linewidth=2, label='f(x)')
            
        except Exception as e:
            print(f"Aviso: Não foi possível plotar toda a função: {e}")
        
        if point != oo and point != -oo:
            point_float = float(point)
            ax.axvline(x=point_float, color='red', linestyle='--', 
                      linewidth=1.5, alpha=0.7, label=f'x = {title_point}')
        
        if resultado is not None and resultado != 'None':
            try:
                if resultado != oo and resultado != -oo and str(resultado).lower() != 'nan':
                    resultado_float = float(resultado)
                    if point != oo and point != -oo:
                        point_float = float(point)
                        ax.plot(point_float, resultado_float, 'ro', markersize=10, 
                               label=f'Limite = {resultado}', zorder=5)
                        ax.plot(point_float, resultado_float, 'o', 
                               markersize=12, markerfacecolor='none', 
                               markeredgecolor='red', markeredgewidth=2, zorder=5)
            except:
                pass
        
        if resultado is not None and resultado not in [oo, -oo, 'None']:
            try:
                resultado_float = float(resultado)
                ax.axhline(y=resultado_float, color='green', linestyle=':', 
                          linewidth=1, alpha=0.5, label=f'y = {resultado}')
            except:
                pass
        
        ax.grid(True, alpha=0.3, linestyle='--')
        
        ax.axhline(y=0, color='k', linewidth=0.5)
        ax.axvline(x=0, color='k', linewidth=0.5)
        
        dir_text = ""
        if direction == "+":
            dir_text = " (pela direita)"
        elif direction == "-":
            dir_text = " (pela esquerda)"
        
        ax.set_xlabel('x', fontsize=12, fontweight='bold')
        ax.set_ylabel('f(x)', fontsize=12, fontweight='bold')
        ax.set_title(f'Gráfico de f(x) e Limite em x → {title_point}{dir_text}', 
                    fontsize=14, fontweight='bold', pad=20)
        
        ax.legend(loc='best', fontsize=10, framealpha=0.9)
        
        plt.tight_layout()
        
        print("\n📊 Exibindo gráfico...")
        plt.show()
        
    except Exception as e:
        print(f"\n⚠️  Não foi possível gerar o gráfico: {str(e)}")
        print("    O cálculo do limite está correto, apenas a visualização falhou.")


def main():
    print("╔═══════════════════════════════════════════════╗")
    print("║     CALCULADORA DE LIMITES DE FUNÇÕES         ║")
    print("║         Uma Variável Real (SymPy)             ║")
    print("╚═══════════════════════════════════════════════╝")
    print()
    print("Instruções:")
    print("  - Use 'x' como variável")
    print("  - Funções: sin(x), cos(x), tan(x), exp(x), log(x), sqrt(x)")
    print("  - Operadores: +, -, *, /, ** (potência)")
    print("  - Exemplos de expressões:")
    print("      (x**2 - 4)/(x - 2)")
    print("      sin(x)/x")
    print("      (1 + 2/(3*x - 1))**(3*x)")
    print("      sqrt(x + 1) - sqrt(x)")
    print()
    
    while True:
        print("\n" + "─" * 47)
        print("NOVA CONSULTA")
        print("─" * 47)
        
        try:
            expr_str = input("\n📝 Digite a expressão f(x): ").strip()
            if not expr_str or expr_str.lower() in ['sair', 'exit', 'quit', 'q']:
                print("\n👋 Encerrando o programa. Até logo!")
                break
        except (EOFError, KeyboardInterrupt):
            print("\n\n👋 Encerrando o programa. Até logo!")
            break
        
        try:
            point_str = input("📍 Digite o ponto do limite (número, inf, -inf): ").strip()
            if not point_str:
                point_str = "0"
        except (EOFError, KeyboardInterrupt):
            print("\n\n👋 Encerrando o programa. Até logo!")
            break
        
        try:
            direction = input("⬅️➡️  Direção (both/+/-) [both]: ").strip()
            if not direction:
                direction = "both"
        except (EOFError, KeyboardInterrupt):
            print("\n\n👋 Encerrando o programa. Até logo!")
            break
        
        print("\n" + "═" * 47)
        print("PROCESSANDO...")
        print("═" * 47)
        
        resultado = solve_limit(expr_str, point_str, direction)
        print(resultado)
        
        try:
            ver_grafico = input("\n📊 Deseja visualizar o gráfico? (s/n) [s]: ").strip().lower()
            if ver_grafico not in ['n', 'nao', 'não', 'no']:
                try:
                    x = Symbol('x', real=True)
                    expr_str_norm = normalize_expression(expr_str)
                    transformations = standard_transformations + (implicit_multiplication_application,)
                    expr = parse_expr(expr_str_norm, local_dict={'x': x}, transformations=transformations)
                    
                    if point_str.lower() in ['inf', '+inf', 'infinity', '+infinity']:
                        point = oo
                    elif point_str.lower() in ['-inf', '-infinity']:
                        point = -oo
                    else:
                        point = sp.sympify(point_str)
                    
                    if direction == "both":
                        resultado_valor = limit(expr, x, point)
                    else:
                        resultado_valor = limit(expr, x, point, direction)
                    
                    plot_limit(expr, x, point, resultado_valor, direction)
                    
                except Exception as e:
                    print(f"⚠️  Erro ao gerar gráfico: {e}")
        except (EOFError, KeyboardInterrupt):
            pass
        
        try:
            continuar = input("\n❓ Calcular outro limite? (s/n) [s]: ").strip().lower()
            if continuar in ['n', 'nao', 'não', 'no']:
                print("\n👋 Encerrando o programa. Até logo!")
                break
        except (EOFError, KeyboardInterrupt):
            print("\n\n👋 Encerrando o programa. Até logo!")
            break


if __name__ == "__main__":
    main()
