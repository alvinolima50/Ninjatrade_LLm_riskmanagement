# Implementação de Text-to-Speech com Autoplay Local

## Modificações Realizadas

Foram feitas as seguintes alterações para implementar a reprodução automática de áudio gerado pelo Text-to-Speech:

1. **Modificações no voice_utils.py**:
   - Adicionado suporte para pygame para reprodução de áudio local
   - Implementada a função `play_audio_locally` que:
     - Cria um arquivo temporário para o áudio
     - Reproduz o áudio usando pygame
     - Limpa o arquivo temporário após a reprodução
   - Atualizada a função `generate_speech_for_response` para:
     - Reproduzir o áudio localmente via pygame
     - Manter o HTML do player para compatibilidade com o navegador

2. **Novo script install_dependencies.py**:
   - Script para instalar o pygame caso não esteja disponível

## Como Funciona

1. Quando o usuário envia uma mensagem no chat, a resposta é gerada como texto
2. O texto é convertido em áudio usando a API Text-to-Speech da OpenAI
3. O áudio é reproduzido automaticamente no sistema local usando pygame
4. O player HTML ainda é exibido no navegador como backup

## Requisitos

- Python 3.x
- pygame (para reprodução de áudio local)
- Bibliotecas existentes do projeto

## Instalação

1. Execute o script de instalação de dependências:
   ```
   python install_dependencies.py
   ```

2. Ou instale manualmente o pygame:
   ```
   pip install pygame
   ```

## Observações

- O áudio é reproduzido automaticamente no sistema, contornando as restrições de autoplay dos navegadores
- Os arquivos temporários de áudio são limpos automaticamente após a reprodução
- Se houver problemas com a reprodução local, o player HTML no navegador ainda estará disponível como fallback
