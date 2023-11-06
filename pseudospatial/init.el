(require 'package)
(let* ((no-ssl (and (memq system-type '(windows-nt ms-dos))
                    (not (gnutls-available-p))))
       (proto (if no-ssl "http" "https")))
  (when no-ssl
    (warn "\
Your version of Emacs does not support SSL connections,
which is unsafe because it allows man-in-the-middle attacks.
There are two things you can do about this warning:
1. Install an Emacs version that does support SSL and be safe.
2. Remove this warning from your init file so you won't see it again."))
  ;; Comment/uncomment these two lines to enable/disable MELPA and MELPA Stable as desired
  (add-to-list 'package-archives (cons "melpa" (concat proto "://melpa.org/packages/")) t)
  ;;(add-to-list 'package-archives (cons "melpa-stable" (concat proto "://stable.melpa.org/packages/")) t)
  (when (< emacs-major-version 24)
    ;; For important compatibility libraries like cl-lib
    (add-to-list 'package-archives (cons "gnu" (concat proto "://elpa.gnu.org/packages/")))))

;; Added by Package.el.  This must come before configurations of
;; installed packages.  Don't delete this line.  If you don't want it,
;; just comment it out by adding a semicolon to the start of the line.
;; You may delete these explanatory comments.
(package-initialize)

(custom-set-variables
 ;; custom-set-variables was added by Custom.
 ;; If you edit it by hand, you could mess it up, so be careful.
 ;; Your init file should contain only one such instance.
 ;; If there is more than one, they won't work right.
 '(TeX-view-program-list '(("MuPDF" ("mupdf %o") "mupdf")))
 '(TeX-view-program-selection
   '(((output-dvi has-no-display-manager)
      "dvi2tty")
     ((output-dvi style-pstricks)
      "dvips and gv")
     (output-dvi "xdvi")
     (output-pdf "MuPDF")
     (output-html "xdg-open")))
 '(ansi-color-faces-vector
   [default default default italic underline success warning error])
 '(custom-enabled-themes '(wombat))
 '(ispell-dictionary "en_US-w_accents")
 '(package-selected-packages '(flycheck company ace-jump-mode expand-region smex auctex))
 '(preview-auto-cache-preamble t))
(custom-set-faces
 ;; custom-set-faces was added by Custom.
 ;; If you edit it by hand, you could mess it up, so be careful.
 ;; Your init file should contain only one such instance.
 ;; If there is more than one, they won't work right.
 '(company-scrollbar-bg ((t (:background "#4a664a664a66"))))
 '(company-scrollbar-fg ((t (:background "#3d993d993d99"))))
 '(company-tooltip ((t (:inherit default :background "#30cc30cc30cc"))))
 '(company-tooltip-common ((t (:inherit font-lock-constant-face))))
 '(company-tooltip-selection ((t (:inherit font-lock-function-name-face)))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;autoconfig/mode-general behaviors;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;; change backup-file directory
(setq backup-directory-alist '(("." . "~/.emacs.d/backup/")))

;; prevent backup by renaming when linked
(setq backup-by-copying-when-linked t)

;; yes-no to y-n
(defalias 'yes-or-no-p 'y-or-n-p)

;; set column width
(setq-default fill-column 100) ; fill-column becomes local when set default 70
(setq whitespace-line-column nil) ; whitespace-mode uses fill-column if whitespace-line-column is nil

;; make searches case sensitive
(setq-default case-fold-search nil)

;; split window below
(setq split-height-threshold nil)
(setq split-width-threshold 9999)

;;enable winner-mode
(winner-mode 1)

;; turn off alarm bell
(setq ring-bell-function 'ignore)

;; make C-j always electric and C-m (RET) always mechanical
(setq electric-indent-mode nil)

;; setup ido and smex
(require 'ido)
(ido-mode t) ; use ido
(setq ido-auto-merge-work-directories-length -1) ; turn off ido-merge directory
(smex-initialize)

;; set ace-jump-search order
(require 'ace-jump-mode)
(setq ace-jump-mode-submode-list
      '(ace-jump-char-mode
	ace-jump-word-mode
	ace-jump-line-mode)) ; the order matches key, C-u key, C-u C-u key

;; setting up the list of recent files
(recentf-mode 1)
(setq recentf-max-saved-items 200)
(setq recentf-max-menu-items 200)

;; enable "confusing commands"
(put 'dired-find-alternate-file 'disabled nil)
(put 'narrow-to-region 'disabled nil)
(put 'narrow-to-page 'disabled nil)
(put 'upcase-region 'disabled nil)
(put 'downcase-region 'disabled nil)
(put 'set-goal-column 'disabled nil)
(put 'scroll-left 'disabled nil)
(put 'scroll-right 'disabled nil)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;mode-specific behaviors;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;; setting up bibtex and reftex under auctex
(setq TeX-parse-self t) ; Enable parse on load.
(setq TeX-auto-save t) ; Enable parse on save.
(setq reftex-bibliography-commands '("bibliography" "nobibliography" "addbibresource"))
(add-hook 'LaTeX-mode-hook 'turn-on-reftex) ; with AUCTeX LaTeX mode
(add-hook 'latex-mode-hook 'turn-on-reftex) ; with Emacs latex mode

;; python indentation
(add-hook 'python-mode-hook
	  #'(lambda () (setq python-indent 2)))

;; setting up line-wrap under org-mode
(require 'org) ; calls "load" to load package "org" when org is not yet loaded
(setq org-startup-truncated nil)

;; auto-started minor-modes in certain major modes
(add-hook 'text-mode-hook 'turn-on-auto-fill)
(add-hook 'latex-mode-hook 'turn-on-auto-fill)
(add-hook 'LaTeX-mode-hook 'turn-on-auto-fill)
(add-hook 'text-mode-hook 'flyspell-mode) ; auto-checking and extra bindings (but the commands are always available)
(add-hook 'latex-mode-hook 'flyspell-prog-mode)
(add-hook 'latex-mode-hook 'flyspell-prog-mode)
(add-hook 'emacs-lisp-mode-hook 'flyspell-prog-mode)
(add-hook 'latex-mode-hook 'flycheck-mode)
(add-hook 'LaTeX-mode-hook 'flycheck-mode)
(add-hook 'emacs-lisp-mode-hook 'flycheck-mode)
(add-hook 'latex-mode-hook 'company-mode)
(add-hook 'LaTeX-mode-hook 'company-mode)
(add-hook 'emacs-lisp-mode-hook 'company-mode)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;x-relevant appearances;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;; transparency
(set-frame-parameter (selected-frame) 'alpha '(100 . 75)) ; (set-frame-parameter (selected-frame) 'alpha '(<active> . <inactive>))
(add-to-list 'default-frame-alist '(alpha . (100 . 75))) ;(set-frame-parameter (selected-frame) 'alpha <both>)

;; move away the mouse when it is potentially confusing
(mouse-avoidance-mode 'exile)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;x-irrelevant appearances;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;;no welcome page
(setq inhibit-startup-screen t)

;;disabling useless bars
(menu-bar-mode -1)
(tool-bar-mode -1)
(scroll-bar-mode -1)
;(fringe-mode nil) ; default of width 8 on both
(fringe-mode '(8 . 0)) ; fringe of width 8 on the left and 0 on the right
;(fringe-mode '(4 . 0)) ; functional minimum
;(fringe-mode -1)  ; no fringes at all
(global-display-line-numbers-mode 1)

;;modeline
(size-indication-mode 1)
(line-number-mode 1)
(column-number-mode 1)
;(setq column-number-indicator-zero-based nil) ; let the column number count from 1
;(setq line-number-display-limit nil) ; always display line-numbers (even when the file has too many lines)
;(setq line-number-display-limit-width 250) ; always display line-numbers (even when the file's lines are too long)
;(display-battery-mode 1) ; taken care by the wm
;(display-time-mode 1)
;(setq display-time-format "%Y%m%d:%A%p:%H:%M")
;(setq display-time-default-load-average 0) ; display 1-minute load

;;highlight the region set by marks
(transient-mark-mode 1)

;;different cursor-appearances in different modes
(defun my-update-cursor ()
  (setq cursor-type (if overwrite-mode 'box '(hbar . 6))))
(add-hook 'buffer-list-update-hook 'my-update-cursor)
(add-hook 'overwrite-mode-hook 'my-update-cursor)

;;numbering in company-mode
(setq company-show-numbers t)

;;set completion-color
(require 'color)
(let ((bg (face-attribute 'default :background)))
  (custom-set-faces
   `(company-scrollbar-bg ((t (:background ,(color-lighten-name bg 15)))))
   `(company-scrollbar-fg ((t (:background ,(color-lighten-name bg 10)))))
   `(company-tooltip ((t (:inherit default :background ,(color-lighten-name bg 5)))))
   `(company-tooltip-common ((t (:inherit font-lock-constant-face))))
   `(company-tooltip-selection ((t (:inherit font-lock-function-name-face))))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;mode-general key-bindings;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;; mac modifiers
(when (eq system-type 'darwin) ;; mac specific settings
  (setq mac-control-modifier 'none)
  (setq mac-command-modifier 'control)
  (setq mac-option-modifier 'meta)
  (setq mac-right-control-modifier 'none)
  (setq mac-right-command-modifier 'control)
  (setq mac-right-option-modifier 'meta)
  (global-set-key [kp-delete] 'delete-char) ;; sets fn-delete to be right-delete
  (setq TeX-view-program-list '(("Preview" ("open -a Preview.app %s.pdf") "Preview")))  ;; set pdf viewer to Preview.app
  (setq TeX-view-program-selection
	'(((output-dvi has-no-display-manager) "dvi2tty")
	  ((output-dvi style-pstricks) "dvips and gv")
	  (output-dvi "xdvi")
	  (output-pdf "Preview.app")
	  (output-html "xdg-open"))))

;;remapping
(global-set-key (kbd "C-?") 'help-command) ; previously C-h
(global-set-key (kbd "M-?") 'mark-paragraph) ; M-h
(global-set-key (kbd "C-h") 'delete-backward-char) ; <backspace>
(global-set-key (kbd "C-S-h") 'kill-whole-line) ; C-S-<backspace>
(global-set-key (kbd "M-h") 'backward-kill-word) ; C-<backspace>
(global-set-key (kbd "C-x M-f") 'find-file-read-only) ; C-x C-r
(global-set-key (kbd "C-.") 'other-window) ; C-x o
(global-set-key (kbd "C-,") (lambda () (interactive) (other-window -1))) ; C-u -1 C-x o
(global-set-key (kbd "C-M-.") 'next-buffer) ; C-x <right>
(global-set-key (kbd "C-M-,") 'previous-buffer) ; C-x <right>
(define-key winner-mode-map (kbd "<backspace>") 'winner-undo) ; C-c <right> ; to preserve <backspace> in global-set-key
(global-set-key (kbd "C-=") 'what-cursor-position) ; C-x =
(global-set-key (kbd "C-:") 'repeat) ; C-x z
(global-set-key (kbd "M-g M-c") 'goto-char) ; M-g c
(global-set-key (kbd "M-g M-g") 'goto-line) ; M-g g
(global-set-key (kbd "C-S-g") 'keyboard-escape-quit) ; ESC ESC ESC
(global-set-key (kbd "M-$") 'flyspell-buffer) ; changing binding for M-symbols commands
(global-set-key (kbd "C-$") 'ispell-word)
(global-set-key (kbd "M-%") nil)
(global-set-key (kbd "C-%") 'query-replace)

;;replace with more powerful/convenient commands
(global-set-key (kbd "C-x C-b") 'ibuffer) ;; ibuffer
(global-set-key (kbd "C-x k") 'kill-this-buffer) ;; ibuffer
(global-set-key (kbd "M-SPC") 'cycle-spacing) ;; a more powerful M-spc, M-/
(global-set-key (kbd "M-/") 'hippie-expand)
(global-set-key (kbd "M-x") 'nil)
(global-set-key (kbd "M-x x") 'execute-extended-command)
(global-set-key (kbd "M-x M-x") 'smex)
(global-set-key (kbd "M-x M-X") 'smex-major-mode-commands)
(global-set-key (kbd "M-g n") 'flyspell-goto-next-error)
(global-set-key (kbd "M-g p") nil)
;(global-set-key (kbd "M-g p") (lambda () (interactive) (flyspell-goto-next-error -1)))
(global-set-key (kbd "M-g N") 'flycheck-next-error)
(global-set-key (kbd "M-g P") 'flycheck-previous-error)

;;binding unbound commands
(global-set-key (kbd "M-n") 'forward-paragraph) ; bind paragraph up/down
(global-set-key (kbd "M-p") 'backward-paragraph)
(global-set-key (kbd "C-x C-r") 'recentf-open-files) ;; recent files
(global-set-key (kbd "C-<") (lambda () (interactive) (scroll-right 40))) ; scrolling laterally
(global-set-key (kbd "C->") (lambda () (interactive) (scroll-left 40)))
;(global-set-key (kbd "C-S-f") (lambda () (interactive) (forward-char 3))) ;; fast forward/backward
;(global-set-key (kbd "C-S-b") (lambda () (interactive) (backward-char 3)))
(global-set-key (kbd "C-M-j") (lambda () (interactive) (move-beginning-of-line 1) (kill-whole-line) (yank) (yank) (previous-line 1))) ;; duplicate line
;(global-set-key (kbd "C-M-j") "\C-a\C- \C-n\M-w\C-y")
(global-set-key (kbd "M-g M-C") 'ace-jump-word-mode) ; C-u ace-jump-mode
(global-set-key (kbd "M-g M-G") 'ace-jump-line-mode) ; C-u C-u ace-jump-mode
;(global-set-key (kbd "<insert>") 'overwrite-mode)
;(global-set-key (kbd "C-<insert>") 'overwrite-mode)
(require 'expand-region)
(global-set-key (kbd "C-`") 'er/expand-region)

;;inserting with <return>
(global-set-key (kbd "<return> g") (lambda (n) (interactive "p") (insert-char #x0025 n))) ; insert percentage
(global-set-key (kbd "<return> q") (lambda (n) (interactive "p") (insert-char #x003D n))) ; equal
(global-set-key (kbd "<return> e <return>") (lambda (n) (interactive "p") (insert-char #x0028 n) (insert-char #x0029 n) (backward-char n)))
;(global-set-key (kbd "<return> e <return>") 'insert-parentheses) ; parenthesis
(global-set-key (kbd "<return> e r") (lambda (n) (interactive "p") (insert-char #x0028 n))) ; left-parenthesis
(global-set-key (kbd "<return> e w") (lambda (n) (interactive "p") (insert-char #x0029 n))) ; right-parenthesis
(global-set-key (kbd "<return> r") (lambda (n) (interactive "p") (insert-char #x002A n))) ; star
(global-set-key (kbd "<return> t") (lambda (n) (interactive "p") (insert-char #x0040 n))) ; at
(global-set-key (kbd "<return> a") (lambda (n) (interactive "p") (insert-char #x0026 n))) ; and
(global-set-key (kbd "<return> s") (lambda (n) (interactive "p") (insert-char #x0023 n))) ; sharp
(global-set-key (kbd "<return> d") (lambda (n) (interactive "p") (insert-char #x0024 n))) ; dollar
(global-set-key (kbd "<return> f") (lambda (n) (interactive "p") (insert-char #x0021 n))) ; factorial
(global-set-key (kbd "<return> x") (lambda (n) (interactive "p") (insert-char #x005E n))) ; exponent
(global-set-key (kbd "<return> p") (lambda (n) (interactive "p") (insert-char #x002B n))) ; plus
(global-set-key (kbd "<return> j") (lambda (n) (interactive "p") (insert-char #x0060 n))) ; grave accent
(global-set-key (kbd "<return> v") (lambda (n) (interactive "p") (insert-char #x005F n))) ; underscore
(global-set-key (kbd "<return> b") (lambda (n) (interactive "p") (insert-char #x002D n))) ; bar
(global-set-key (kbd "<return> w") (lambda (n) (interactive "p") (insert-char #x007E n))) ; tilde
(global-set-key (kbd "<return> i") (lambda (interactive) (insert-char #x005E) (insert-char #x007B) (insert-char #x002D) (insert-char #x0031) (insert-char #x007D)))

;;personal commands with prefix C-z and C-;
(global-set-key (kbd "C-z") nil) ; unbind C-z (suspend-emacs is also bound to C-x C-z)
(global-set-key (kbd "C-z C-c") 'clone-buffer)
(global-set-key (kbd "C-z C-e") 'eshell)
(global-set-key (kbd "C-z C-r") 'rename-uniquely)
(global-set-key (kbd "C-z C-s") 'shell)
(global-set-key (kbd "C-z C-t") 'toggle-truncate-lines)
(global-set-key (kbd "C-z C-b") (lambda (n) (interactive "p") ; add {} to region
				  (insert-char #x007B n) (exchange-point-and-mark) (insert-char #x007D n)))
(global-set-key (kbd "C-z C-v") (lambda (n) (interactive "p") ; remove symbols from ends
				  (delete-char 1) (exchange-point-and-mark) (backward-char 1) (delete-char 1)))
(global-set-key (kbd "C-;") nil)
(global-set-key (kbd "C-; C-h") 'hl-line-mode)
(global-set-key (kbd "C-; C-n") (lambda () (interactive) (whitespace-mode 'toggle))) ; update fill-column for whitespace-mode
(global-set-key (kbd "C-; C-j") 'ace-jump-mode)
(global-set-key (kbd "C-; C-k") 'ace-jump-mode-pop-mark)
;(global-set-key (kbd "C-; C-m") 'calc)
(require 'dired)
(global-set-key (kbd "C-z M o") 'org-mode)
(global-set-key (kbd "C-z M t") 'text-mode)
(global-set-key (kbd "C-z m c") 'company-mode)
(global-set-key (kbd "C-z m d") 'toggle-debug-on-error)
(global-set-key (kbd "C-z m f") 'auto-fill-mode)
(global-set-key (kbd "C-z m l") 'follow-mode)
(global-set-key (kbd "C-z m p") 'flyspell-prog-mode)
(global-set-key (kbd "C-z m s") 'flyspell-mode)
(global-set-key (kbd "C-z m y") 'flycheck-mode)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;mode-specific key-bindings;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;;isearch
(define-key isearch-mode-map (kbd "C-?") 'help-command)
(define-key isearch-mode-map (kbd "M-?") 'mark-paragraph)
(define-key isearch-mode-map (kbd "C-h") 'isearch-del-char)
(define-key isearch-mode-map (kbd "C-w") nil)
(define-key isearch-mode-map (kbd "C-=") 'isearch-yank-word-or-char)
(define-key isearch-mode-map (kbd "<return>") nil)

;;company-mode
(defvar company-active-map
  (let ((keymap (make-sparse-keymap)))
    (define-key keymap "\e\e\e" 'company-abort)
    (define-key keymap "\C-g" 'company-abort)
    (define-key keymap (kbd "C-S-n") 'company-select-next)
    (define-key keymap (kbd "C-S-p") 'company-select-previous)
    (define-key keymap (kbd "C-S-v") 'company-next-page)
    (define-key keymap (kbd "M-V") 'company-previous-page)
    (define-key keymap (kbd "C-S-m") 'company-complete-selection)
    (define-key keymap (kbd "C-i") 'company-complete-common)
    (define-key keymap (kbd "<f1> f") 'company-show-doc-buffer)
    (define-key keymap (kbd "C-? f") 'company-show-doc-buffer)
    (define-key keymap (kbd "M-.") 'company-show-location)
    (define-key keymap (kbd "C-S-s") 'company-search-candidates)
    (define-key keymap (kbd "M-S") 'company-filter-candidates)
    (dotimes (i 10)
      (define-key keymap (read-kbd-macro (format "<return> %d" i)) 'company-complete-number))
     keymap)
  "Keymap that is enabled during an active completion.")

;;org-mode
(define-key org-mode-map (kbd "<tab>") nil) ; org-cycle
(define-key org-mode-map (kbd "C-i") nil)
(define-key org-mode-map (kbd "<right>") 'org-cycle)
(define-key org-mode-map (kbd "S-<iso-lefttab>") nil) ; org-shifttab
(define-key org-mode-map (kbd "S-<tab>") nil)
(define-key org-mode-map (kbd "<backtab>") nil)
(define-key org-mode-map (kbd "<left>") 'org-shifttab)
(define-key org-mode-map (kbd "C-<tab>") nil) ; org-force-cycle-archived
(define-key org-mode-map (kbd "C-<right>") 'org-force-cycle-archived)
(define-key org-mode-map (kbd "C-<left>") nil) ; left-word
(define-key org-mode-map (kbd "C-M-<left>") nil) ; backward-sexp
(define-key org-mode-map (kbd "C-M-<right>") nil) ; forward-sexp
(define-key org-mode-map (kbd "<up>") nil) ; previous-line
(define-key org-mode-map (kbd "C-<up>") nil) ; org-backward-paragraph
(define-key org-mode-map (kbd "<down>") nil) ; previous-line
(define-key org-mode-map (kbd "C-<down>") nil) ; org-backward-paragraph
(define-key org-mode-map (kbd "M-h") nil) ; org-mark-element
(define-key org-mode-map (kbd "M-?") 'org-mark-element)
(define-key org-mode-map (kbd "C-,") nil) ; org-cycle-agenda-files
(define-key org-mode-map (kbd "C-.") nil) ; org-cycle-agenda-files


;;flyspell-mode
(require 'flyspell)
(define-key flyspell-mode-map (kbd "C-c $") nil) ; some weird float-window interfaces
(define-key flyspell-mode-map (kbd "C-.") nil) ; auto-correcting next
(define-key flyspell-mode-map (kbd "C-M-i") nil) ; correct at point
(define-key flyspell-mode-map (kbd "C-;") nil) ; auto-correcting previous
(define-key flyspell-mode-map (kbd "C-,") nil) ; go to next

;;flymake-mode
(setq flycheck-keymap-prefix (kbd "C-c <return> f"))

;;latex mode
;(define-key (current-local-map) key nil)
;
;(with-eval-after-load "latex"
;  (define-key LaTeX-mode-map (kbd "C-c e")
;    (lambda ()
;      (interactive)
;      (LaTeX-environment 1))))
(require 'latex)
;(require 'tex-mode)
(define-key LaTeX-mode-map (kbd "M-q") 'fill-paragraph)

;;reftex mode
(require 'reftex)
(define-key reftex-mode-map (kbd "C-c <return> a") 'reftex-view-crossref)
(define-key reftex-mode-map (kbd "C-c <return> e r") 'reftex-label)
(define-key reftex-mode-map (kbd "C-c <return> e w") 'reftex-reference)
(define-key reftex-mode-map (kbd "C-c <return> b") 'reftex-toc-recenter)
;(define-key reftex-mode-map (kbd "C-c /") 'reftex-index-selection-or-word)
;(define-key reftex-mode-map (kbd "C-c <") 'reftex-index)
(define-key reftex-mode-map (kbd "C-c <return> w") 'reftex-toc)
;(define-key reftex-mode-map (kbd "C-c >") 'reftex-display-index)
;(define-key reftex-mode-map (kbd "C-c [") 'reftex-citation)
;(define-key reftex-mode-map (kbd "C-c \\") 'reftex-index-phrase-selection-or-word)
;(define-key reftex-mode-map (kbd "C-c |") 'reftex-index-visit-phrases-buffer)

;;ido mode (by adding a hooked function)
(defun bind-ido-keys ()
  "Keybindings for ido mode."
  (define-key ido-completion-map (kbd "C-.") 'other-window) ; restore
  (define-key ido-completion-map (kbd "SPC") (lambda (n) (interactive "p") (insert-char #x0020 n)))
  (define-key ido-completion-map (kbd "S-SPC") 'ido-complete-space))
(add-hook 'ido-setup-hook #'bind-ido-keys)

;;smex mode (by overwriting the problematic function)
(defun smex-prepare-ido-bindings ()
  (define-key ido-completion-map (kbd "M-.") 'smex-find-function) ; find the code that implements the selected function ; restore the problematic function's original functionality
  (define-key ido-completion-map (kbd "C-a") 'move-beginning-of-line)
  (define-key ido-completion-map (kbd "C-i") 'minibuffer-complete) ; change TAB to C-i to avoid potential problems induced by keyboard-changing
  (define-key ido-completion-map (kbd "C-? f") 'smex-describe-function) ; describe selected function ; unbinding C-h for delete-backward-char
  (define-key ido-completion-map (kbd "C-? w") 'smex-where-is) ; show bindings of the selected function
  (define-key ido-completion-map (kbd "C-.") 'other-window) ; new functionalities
  (define-key ido-completion-map (kbd "SPC") (lambda (n) (interactive "p") (insert-char #x0020 n)))
  (define-key ido-completion-map (kbd "S-SPC") 'ido-complete-space))
