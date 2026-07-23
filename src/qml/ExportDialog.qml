import QtQuick 2.12
import QtQuick.Controls 2.12
import QtQuick.Controls.Material 2.12
import QtQuick.Layouts 1.12
import Qt.labs.platform 1.1

// Opened from the ribbon's "Export" button. Lets the user pick one or more
// .raw recordings from disk and convert each to CSV (CD events) or an .mp4
// via exportWorker/exportEvents (see gui/export_worker.h) -- both global
// context properties set once in combined_main.cpp, the same way
// cameraWorker/cameraEvents are, so no plumbing through Ribbon/Main.qml is
// needed here.
Popup {
    id: root
    modal: true
    focus: true
    closePolicy: Popup.CloseOnEscape
    width: 460
    height: 480
    anchors.centerIn: Overlay.overlay

    property bool exporting: false
    // Captured at selection time rather than re-read from rawFileDialog.files
    // later -- the dialog's own file-list property is meant to reflect its
    // last selection, not to be held onto as persistent state.
    property var selectedFileUrls: []

    Theme { id: theme }

    function localPathFromUrl(url) {
        var s = url.toString()
        if (s.indexOf("file://") === 0) {
            s = s.substring(7)
        }
        return decodeURIComponent(s)
    }

    ListModel { id: fileModel }
    ListModel { id: logModel }

    FileDialog {
        id: rawFileDialog
        title: "Select .raw recordings"
        fileMode: FileDialog.OpenFiles
        nameFilters: ["RAW recordings (*.raw)", "All files (*)"]
        onAccepted: {
            // Copy each url out as a plain string (via toString()) rather
            // than holding onto the `files` array/elements themselves --
            // that array is owned by the native dialog and goes back to
            // empty once the dialog resets, which silently emptied
            // selectedFileUrls by the time Start Export was clicked.
            var urls = []
            fileModel.clear()
            for (var i = 0; i < files.length; ++i) {
                var urlString = files[i].toString()
                urls.push(urlString)
                var filePath = root.localPathFromUrl(urlString)
                fileModel.append({
                    filePath: filePath,
                    fileLabel: filePath.split("/").pop(),
                    conversionStatus: "pending"
                })
            }
            root.selectedFileUrls = urls
        }
    }

    Connections {
        target: exportEvents
        onLogLine: {
            logModel.append({ lineText: line })
            logView.positionViewAtEnd()
        }
        onFileStarted: {
            for (var i = 0; i < fileModel.count; ++i) {
                if (fileModel.get(i).filePath === path) {
                    fileModel.setProperty(i, "conversionStatus", "converting")
                }
            }
        }
        onFileFinished: {
            for (var i = 0; i < fileModel.count; ++i) {
                if (fileModel.get(i).filePath === path) {
                    fileModel.setProperty(i, "conversionStatus", success ? "done" : "failed")
                }
            }
        }
        onExportFinished: {
            root.exporting = false
        }
    }

    contentItem: ColumnLayout {
        spacing: 10

        Label {
            text: "Export Recordings"
            font.bold: true
            font.pixelSize: 16
            color: theme.text
        }

        RowLayout {
            Layout.fillWidth: true
            spacing: 10

            Button {
                text: "Choose .raw Files..."
                enabled: !root.exporting
                onClicked: rawFileDialog.open()
            }
            Label {
                text: fileModel.count > 0 ? fileModel.count + " file(s) selected" : "No files selected"
                color: theme.mutedText
            }
        }

        ListView {
            Layout.fillWidth: true
            Layout.preferredHeight: 110
            clip: true
            model: fileModel
            delegate: RowLayout {
                width: ListView.view.width
                Label {
                    text: fileLabel
                    color: theme.text
                    elide: Text.ElideMiddle
                    Layout.fillWidth: true
                }
                Label {
                    text: conversionStatus === "converting" ? "…" :
                          conversionStatus === "done" ? "✓" :
                          conversionStatus === "failed" ? "✗" : ""
                    color: conversionStatus === "failed" ? theme.danger : theme.accent
                }
            }
        }

        RowLayout {
            spacing: 16

            RadioButton {
                id: csvRadio
                text: "CSV (CD events)"
                checked: true
                enabled: !root.exporting
            }
            RadioButton {
                id: videoRadio
                text: "Video (.mp4)"
                enabled: !root.exporting
            }
        }

        CheckBox {
            id: monochromeCheck
            text: "Black and White"
            visible: videoRadio.checked
            enabled: !root.exporting
        }

        Label {
            Layout.fillWidth: true
            wrapMode: Text.WordWrap
            color: theme.mutedText
            font.pixelSize: 11
            text: "Each output is written next to its source .raw, same name" +
                  (videoRadio.checked && monochromeCheck.checked ? " plus \"_BW\"" : "") +
                  ", with a ." + (videoRadio.checked ? "mp4" : "csv") + " extension."
        }

        ListView {
            id: logView
            Layout.fillWidth: true
            Layout.fillHeight: true
            clip: true
            model: logModel
            delegate: Label {
                width: ListView.view.width
                text: lineText
                color: theme.mutedText
                font.family: "monospace"
                font.pixelSize: 11
                wrapMode: Text.WrapAnywhere
            }
        }

        RowLayout {
            Layout.fillWidth: true

            Button {
                text: root.exporting ? "Exporting..." : "Start Export"
                enabled: fileModel.count > 0 && !root.exporting
                highlighted: true
                onClicked: {
                    root.exporting = true
                    logModel.clear()
                    for (var i = 0; i < fileModel.count; ++i) {
                        fileModel.setProperty(i, "conversionStatus", "pending")
                    }
                    exportWorker.exportFiles(root.selectedFileUrls, videoRadio.checked ? "video" : "csv",
                                              monochromeCheck.checked)
                }
            }
            Item { Layout.fillWidth: true }
            Button {
                text: "Close"
                onClicked: root.close()
            }
        }
    }
}
